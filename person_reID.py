'''
Person Re-IDentification

Ref: None (for my own implementation)
'''
import os
from scipy.io import loadmat
from datetime import datetime
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from imageio import imread
from tqdm import tqdm

import torch
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from torchvision_utils.engine import train_one_epoch, evaluate
from torchvision_utils import utils

from models import pretrained_model
from custom_data.dataset import CUHKSYSUDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu cuda index')
    parser.add_argument('--data_dir', type=str, default='dataset/CUHK-SYSU/', help='data directory')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--pretrained_model', type=str, default='model_epoch49_2021-06-12-19-06-07.pth', help='pretrained model name')
    parser.add_argument('--evaluate', action='store_true', default=False, help='only evaluate')
    args = parser.parse_args()
    return args

def get_train_test_images(data_dir):
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


def main(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)  # change allocation of current GPU
    print('Current cuda device ', torch.cuda.current_device())  # check

    dataset = CUHKSYSUDataset(args.data_dir, train=True)
    dataset_test = CUHKSYSUDataset(args.data_dir, train=False)

    train_indices, test_indices = get_train_test_images(args.data_dir)

    dataset = Subset(dataset, train_indices)
    dataset_test = Subset(dataset_test, test_indices)

    data_loader = DataLoader(dataset, batch_size=2, shuffle=False,
                             num_workers=4, collate_fn=utils.collate_fn)

    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False,
                                  num_workers=0, collate_fn=utils.collate_fn)

    model = pretrained_model.load_pretrained_faster_rcnn()
    checkpoint = torch.load('checkpoints/fasterrcnn/model_epoch49_2021-06-12-19-06-07.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)


    print('LOAD and EVALUATE! ================')
    #####simclr_model
    print('SimCLR')
    simclr_model = pretrained_model.load_trained_simclr_model(device)
    simclr_model.eval()
    ids = list(os.listdir(args.data_dir + 'Image/bbox/Train_only1'))
    print('ids length', len(ids))
    dataset_target = D.ImageFolder(args.data_dir + 'Image/bbox/Train_only1',
                                   transform=T.Compose([T.Resize((250, 80)), T.ToTensor()]))
    data_loader_target = DataLoader(dataset_target, batch_size=64, shuffle=True,
                                    num_workers=0, collate_fn=utils.collate_fn)
    target_reps = torch.empty((0, 128)).to(device)
    for images, target in data_loader_target:
        images = torch.stack(images).to(device)
        target_rep = simclr_model(images)
        target_reps = torch.cat((target_reps, target_rep), dim=0)

    ## faster r-cnn
    print('Faster R-CNN')
    model.eval()
    img_ids = list(sorted(os.listdir(args.data_dir + 'Image/SSM')))
    out_reps = torch.empty((0, 128)).to(device)
    img_names = []
    for image, target in tqdm(data_loader_test):
        skip_flag = False
        image = list(img.to(device) for img in image)
        target = target[0]
        img_name = img_ids[target['image_id']]
        gt_boxes = target['boxes']
        output = model(image)
        #print('image name {} # of boxes {}'.format(img_name, target['num'].item()))
        out_boxes = output[0]['boxes']
        #print(gt_boxes.size(), out_boxes.size())
        if out_boxes.size()[0] > 0:
            img_names.append(img_name)
            crops = torch.empty((0,3,250,85))
            img = imread(args.data_dir + 'Image/SSM/' + img_name)
            for i in range(min(target['num'],out_boxes.size()[0])):
                box = list(map(int, out_boxes[i,:]))
                crop = img[box[0]:box[2],box[1]:box[3]]
                if crop.shape[0] * crop.shape[1] * crop.shape[2] == 0:
                    skip_flag = True
                    continue
                crop = crop.reshape(3, crop.shape[0], crop.shape[1])
                crop = torch.unsqueeze(torch.Tensor(crop), 0)
                crop = F.interpolate(crop, (250,85))
                crops = torch.cat((crops, crop), dim=0)
            if not skip_flag:
                crops = crops.to(device)
                out_rep = simclr_model(crops)
                out_reps = torch.cat((out_reps, out_rep), dim=0)

    ## similarity matching
    print('Similarity')
    target_reps = target_reps[~torch.any(target_reps.isnan(), dim=1)]
    target_reps = target_reps[~torch.any(target_reps.isinf(), dim=1)]
    out_reps = out_reps[~torch.any(out_reps.isnan(), dim=1)]
    out_reps = out_reps[~torch.any(out_reps.isinf(), dim=1)]
    similarity = cosine_similarity(out_reps.cpu().data.numpy(), target_reps.cpu().data.numpy())
    print(similarity.shape)
    indices = torch.argmax(torch.Tensor(similarity), dim=1).tolist()
    result_ids = list(np.array(ids)[indices])
    with open("checkpoints/results/result-{}.txt".format(datetime.now().isoformat().replace('T', '-').replace(':', '-')[:-7]), 'w') as f:
        for s in result_ids:
            f.write(str(s) + '\n')
    return result_ids

    print('evaluate')
    evaluate(model, data_loader, device=device)

    print('test testset')
    evaluate(model, data_loader_test, device=device)


if __name__ == '__main__':
    args = get_args()
    result_ids = main(args)

