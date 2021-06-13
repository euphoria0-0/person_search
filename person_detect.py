'''
Person Detection using Faster R-CNN

Source:
    Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
    PyTorch

Reference:
    https://arxiv.org/abs/1506.01497
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
'''
import os
from scipy.io import loadmat
from imageio import imread
from datetime import datetime
import argparse
import cv2
import torch
from torch.utils.data import DataLoader, Subset

from torchvision_utils.engine import train_one_epoch, evaluate
from torchvision_utils import utils
from custom_data.dataset import CUHKSYSUDataset
from models import pretrained_model


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
    if args.pretrained_model is not None:
        #checkpoint = torch.load('checkpoints/model_epoch49_2021-06-12-19-06-07.pth')
        checkpoint = torch.load('checkpoints/fasterrcnn/'+args.pretrained_model)
        model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    if not args.evaluate:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        print('Train! ================')
        for epoch in range(50, args.num_epochs):
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
            lr_scheduler.step()
            # print('evaluate start')
            # evaluate(model, data_loader_test, device=device)
            # print('evaluate end')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'checkpoints/fasterrcnn/model_epoch{}_{}.pth'.format(epoch,
                                                         datetime.now().isoformat().replace('T', '-').replace(':', '-')[:-7]))

    print('LOAD and EVALUATE! ================')
    model.eval()
    img_ids = list(sorted(os.listdir(args.data_dir + 'Image/SSM')))
    for image, target in data_loader:
        image = list(img.to(device) for img in image)
        target = target[0]
        img_name = img_ids[target['image_id']]
        gt_boxes = target['boxes']
        output = model(image)
        print('image name {} # of boxes {}'.format(img_name, target['num'].item()))
        out_boxes = output[0]['boxes']

        img = imread(args.data_dir + 'Image/SSM/' + img_name)
        for i in range(target['num']):
            box = out_boxes[i, :]
            gt_box = gt_boxes[i, :]
            cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (0, 0, 255), 2)
            cv2.rectangle(img, (gt_box[1], gt_box[0]), (gt_box[3], gt_box[2]), (255, 0, 0), 2)
        cv2.imwrite('bbox/img_{}_{}boxes_{}.jpg'.format(img_name.replace('.jpg', ''), target['num'].item(),
                                                        datetime.now().isoformat().replace('T', '-').replace(':','-')),
                    img)

        break

    print('evaluate')
    evaluate(model, data_loader, device=device)

    print('test testset')
    evaluate(model, data_loader_test, device=device)


if __name__ == '__main__':
    args = get_args()
    result_ids = main(args)

