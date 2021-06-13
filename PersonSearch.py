'''
Person Search

Implementation Reference
    Person Detection
        Faster R-CNN & DataSet class
        https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    Person Re-Identification
        None (Our own implementation)

Before you run this file,
1. download CUHK-SYSU dataset and unzip it in dataset folder.
2. run simclr_training.py for training SimCLR(representation learner).
'''
import os
import argparse
import cv2
import numpy as np
from imageio import imread
from scipy.io import loadmat
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F

from torchvision_utils.engine import train_one_epoch, evaluate
from torchvision_utils import utils
from custom_data.dataset import CUHKSYSUDataset
from custom_data.data_preprocessing import data_preprocessing
from models import pretrained_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu cuda index')
    parser.add_argument('--data_dir', type=str, default='dataset/CUHK-SYSU/', help='data directory')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--pretrained_simclr', type=str, default='checkpoint_1000.pth.tar',
                        help='pretrained SimCLR model file name')
    parser.add_argument('--pretrained_fasterrcnn', type=str, default='model_epoch49_2021-06-12-19-06-07.pth',
                        help='pretrained Faster R-CNN model file name')
    parser.add_argument('--evaluate', action='store_true', default=False, help='only evaluate')
    args = parser.parse_args()
    return args

def get_train_test_images(data_dir):
    # split train and test data
    # 5532 query persons for training
    train = loadmat(data_dir + 'annotation/test/train_test/Train.mat')
    train_data = []
    for i in range(len(train['Train'])):
        x = train['Train'][i][0][0][0]
        for xx in x[2][0]:
            train_data.append(xx[0][0])

    train_data = list(set(train_data))

    test = loadmat(data_dir + 'annotation/pool.mat')
    test_data = [x[0][0] for x in test['pool']]

    images_order = list(sorted(os.listdir(data_dir + 'Image/SSM')))

    train_indices = []
    for train_img in train_data:
        train_indices.append(images_order.index(train_img))

    test_indices = []
    for test_img in test_data:
        test_indices.append(images_order.index(test_img))

    # random.shuffle(train_indices)
    # random.shuffle(test_indices)
    return train_indices, test_indices

def draw_detection_results(model, data_loader, device, data_dir):
    model.eval()
    # draw image and bounding boxes to check performance of object detection
    img_ids = list(sorted(os.listdir(data_dir + 'Image/SSM')))
    for image, target in data_loader:
        image = list(img.to(device) for img in image)
        target = target[0]
        img_name = img_ids[target['image_id']]
        gt_boxes = target['boxes']
        output = model(image)
        print('image name {} # of boxes {}'.format(img_name, target['num'].item()))
        out_boxes = output[0]['boxes']
        # draw bboxes
        img = imread(data_dir + 'Image/SSM/' + img_name)
        for i in range(target['num']):
            box = out_boxes[i, :]
            gt_box = gt_boxes[i, :]
            cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (0, 0, 255), 2)
            cv2.rectangle(img, (gt_box[1], gt_box[0]), (gt_box[3], gt_box[2]), (255, 0, 0), 2)

        now = datetime.now().isoformat().replace('T', '-').replace(':', '-')
        cv2.imwrite(data_dir + 'Image/SSM/bbox/img_{}_{}boxes_{}.jpg'.format(img_name.replace('.jpg', ''), target['num'].item(), now), img)
        break  # check only one image


def main(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)  # change allocation of current GPU
    print('Current cuda device ', torch.cuda.current_device())  # check

    ## 1. Data preparation
    
    data_preprocessing(args.data_dir)

    dataset = CUHKSYSUDataset(args.data_dir, train=True)
    dataset_test = CUHKSYSUDataset(args.data_dir, train=False)

    train_indices, test_indices = get_train_test_images(args.data_dir)

    dataset = Subset(dataset, train_indices)
    dataset_test = Subset(dataset_test, test_indices)

    data_loader = DataLoader(dataset, batch_size=2, shuffle=False,
                             num_workers=4, collate_fn=utils.collate_fn)

    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False,
                                  num_workers=0, collate_fn=utils.collate_fn)

    ### 2. Person Detection

    # make model (Faster R-CNN)
    model = pretrained_model.load_pretrained_faster_rcnn()
    if args.pretrained_model is not None:
        checkpoint = torch.load('checkpoints/fasterrcnn/'+args.pretrained_fasterrcnn)
        model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    ## train Faster R-CNN
    print('Training Faster R-CNN')
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(args.num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
        lr_scheduler.step()
        # print('evaluate start')
        # evaluate(model, data_loader_test, device=device)
        # print('evaluate end')
        now = datetime.now().isoformat().replace('T', '-').replace(':', '-')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'checkpoints/fasterrcnn/model_epoch{}_{}.pth'.format(epoch, now))
        if epoch % 10 == 0 and epoch > 0:
            print('Evaluation on epoch {}'.format(epoch))
            evaluate(model, data_loader_test, device=device)

    ## evaluate Faster R-CNN
    # show results of object detection
    draw_detection_results(model, data_loader, device, args.data_dir)

    print('Test Faster R-CNN')
    # compute mAP
    evaluate(model, data_loader_test, device=device)


    ### 3. Person Re-Identification

    # load SimCLR model
    simclr_model = pretrained_model.load_trained_simclr_model(device, args.pretrained_simclr)
    simclr_model.eval()

    # inference of target images using SimCLR
    # input: target images
    # output: representations of target images
    print('SimCLR')
    ids = list(os.listdir(args.data_dir + 'Image/bbox/Train_only1'))
    print('ids length', len(ids))
    dataset_target = D.ImageFolder(args.data_dir + 'Image/bbox/Train_only1',
                                   transform=T.Compose([T.Resize((250, 80)), T.ToTensor()]))
    data_loader_target = DataLoader(dataset_target, batch_size=64, shuffle=True,
                                    num_workers=0, collate_fn=utils.collate_fn)
    target_reps = torch.empty((0, 128)).to(device)
    for images, target in data_loader_target:
        images = torch.stack(images).to(device)
        target_rep = simclr_model(images)  # inference
        target_reps = torch.cat((target_reps, target_rep), dim=0)

    # prediction of detected images (by Faster R-CNN) using SimCLR
    # input: detected images
    # output: representations of detected images
    print('Faster R-CNN')
    img_ids = list(sorted(os.listdir(args.data_dir + 'Image/SSM')))
    out_reps = torch.empty((0, 128)).to(device)
    img_names = []
    for image, target in tqdm(data_loader_test):
        skip_flag = False
        image = list(img.to(device) for img in image)
        target = target[0]
        img_name = img_ids[target['image_id']]
        output = model(image)
        print('image name {} # of boxes {}'.format(img_name, target['num'].item()))
        out_boxes = output[0]['boxes']
        if out_boxes.size()[0] > 0:
            img_names.append(img_name)
            crops = torch.empty((0, 3, 250, 85))
            img = imread(args.data_dir + 'Image/SSM/' + img_name)
            for i in range(min(target['num'], out_boxes.size()[0])):
                box = list(map(int, out_boxes[i, :]))  # predicted bounding box
                crop = img[box[0]:box[2], box[1]:box[3]]  # cropped image
                if crop.shape[0] * crop.shape[1] * crop.shape[2] == 0:
                    skip_flag = True
                    continue
                crop = crop.reshape(3, crop.shape[0], crop.shape[1])
                crop = torch.unsqueeze(torch.Tensor(crop), 0)
                crop = F.interpolate(crop, (250, 85))  # resize cropped image
                crops = torch.cat((crops, crop), dim=0)  # stack them
            if not skip_flag:
                crops = crops.to(device)
                out_rep = simclr_model(crops)
                out_reps = torch.cat((out_reps, out_rep), dim=0)

    ## matching similarity
    # input: representations of target persons and detected persons
    # output: similarity matrix and predicted person ID
    print('Similarity')
    # representations of target persons
    target_reps = target_reps[~torch.any(target_reps.isnan(), dim=1)]
    target_reps = target_reps[~torch.any(target_reps.isinf(), dim=1)]
    # representations of detected persons from gallery images
    out_reps = out_reps[~torch.any(out_reps.isnan(), dim=1)]
    out_reps = out_reps[~torch.any(out_reps.isinf(), dim=1)]
    # compute cosine similarity
    similarity = cosine_similarity(out_reps.cpu().data.numpy(), target_reps.cpu().data.numpy())
    print(similarity.shape)
    # find the most similar (max cosine sim value) person
    indices = torch.argmax(torch.Tensor(similarity), dim=1).tolist()
    # save predicted IDs
    result_ids = list(np.array(ids)[indices])
    with open("checkpoints/results/result-{}.txt".format(
            datetime.now().isoformat().replace('T', '-').replace(':', '-')[:-7]), 'w') as f:
        for s in result_ids:
            f.write(str(s) + '\n')
    # evaluate results: mAP
    evaluate(model, data_loader, device=device)
    # evaluate test results: mAP
    print('TEST Step ===============')
    evaluate(model, data_loader_test, device=device)


if __name__ == '__main__':
    args = get_args()
    result_ids = main(args)

