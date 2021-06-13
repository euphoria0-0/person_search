'''
Person Re-IDentification

Ref: None (for my own implementation)
'''
import os
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torchvision.transforms as T
import torchvision.datasets as D
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast

from torchvision_utils.engine import train_one_epoch, evaluate
from torchvision_utils import utils

from models import pretrained_model
from custom_data.dataset import CUHKSYSUDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu cuda index')
    parser.add_argument('--data_dir', type=str, default='../dps/dataset/CUHK-SYSU/', help='data directory')
    #parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--pretrained_model', type=str, default='model_epoch49_2021-06-12-19-06-07.pth', help='pretrained model name')
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)  # change allocation of current GPU
    print('Current cuda device ', torch.cuda.current_device())  # check

    simclr_model = pretrained_model.load_trained_simclr_model(device, model_name='checkpoint_1000.pth.tar')
    simclr_model.eval()

    ## target
    target_size = [3, 250, 85]
    ids = list(os.listdir(args.data_dir + 'Image/bbox/Train_only1'))
    print('number of IDs of target persons', len(ids))
    dataset_target = D.ImageFolder(args.data_dir + 'Image/bbox/Train_only1',
                                   transform=T.Compose([T.Resize(target_size[1:]), T.ToTensor()]))
    data_loader_target = DataLoader(dataset_target, batch_size=64, shuffle=False,
                                    num_workers=0, collate_fn=utils.collate_fn)

    target_features = torch.empty((0,128)).to(device)
    for imgs, target in tqdm(data_loader_target): # mean: [3, 241.69, 84.64]
        # images = torch.cat(images, dim=0)
        images = torch.empty([0]+target_size)
        for img in imgs:
            img = torch.unsqueeze(img, dim=0)
            images = torch.cat((images, img), dim=0)
        images = images.to(device)

        with torch.no_grad():
            with autocast(enabled=False):
                features = simclr_model(images)
                target_features = torch.cat((target_features, features), dim=0)


    ## test
    test_size = [3, 250, 85]
    test_ids = list(os.listdir(args.data_dir + 'Image/bbox/TestG50'))
    print('number of IDs of persons in test gallery images', len(test_ids))
    dataset_test = D.ImageFolder(args.data_dir + 'Image/bbox/TestG50',
                                   transform=T.Compose([T.Resize(test_size[1:]), T.ToTensor()]))
    data_loader_test = DataLoader(dataset_test, batch_size=64, shuffle=False,
                                    num_workers=0, collate_fn=utils.collate_fn)

    test_features = torch.empty((0, 128)).to(device)
    for imgs, _ in tqdm(data_loader_test):  # mean: 243.58 84.40379310344828
        images = torch.empty([0] + test_size)
        for img in imgs:
            img = torch.unsqueeze(img, dim=0)
            images = torch.cat((images, img), dim=0)
        images = images.to(device)

        with torch.no_grad():
            with autocast(enabled=False):
                features = simclr_model(images)
                test_features = torch.cat((test_features, features), dim=0)


    ## similarity
    test_features = test_features[~torch.any(test_features.isnan(), dim=1)]
    test_features = test_features[~torch.any(test_features.isinf(), dim=1)]

    target_features = target_features[~torch.any(target_features.isnan(), dim=1)]
    target_features = target_features[~torch.any(target_features.isinf(), dim=1)]

    similarity = cosine_similarity(test_features.cpu().data.numpy(), target_features.cpu().data.numpy())
    print('similarity matrix shape {}'.format(similarity.shape))

    indices = torch.argmax(torch.Tensor(similarity), dim=1).tolist()
    result_ids = list(np.array(ids)[indices])

    acc = 0
    with open("checkpoints/results/result-{}.txt".format(datetime.now().isoformat().replace('T', '-').replace(':', '-')[:-7]), 'w') as f:
        for x, y in zip(test_ids, result_ids):
            f.write('{},{}\n'.format(x, y))
            if x == y:
                acc += 1

    print('ACC {:4f}'.format(acc/len(result_ids)))



if __name__ == '__main__':
    args = get_args()
    result_ids = main(args)

