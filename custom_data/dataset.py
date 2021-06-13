'''
Data preparation

Source:
    PyTorch

Reference:
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
'''
import os
from scipy.io import loadmat
from PIL import Image
import pandas as pd
import torch
import torchvision.transforms as T


class CUHKSYSUDataset(object):
    def __init__(self, root, train=True):
        self.root = root
        transforms = [T.ToTensor()]
        if train:
            transforms.append(T.RandomHorizontalFlip())
        self.transforms = T.Compose(transforms)
        self.imgs = list(sorted(os.listdir(self.root + 'Image/SSM')))
        self.train = train

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.root + 'Image/SSM', img_name)
        img = Image.open(img_path).convert("RGB")

        img_file = pd.read_csv(self.root + 'processed/images.csv')
        image = img_file.loc[img_file['imname'] == img_name]
        num_objs = int(image['nAppear'])
        image_id = image.index
        image_id2 = self.imgs.index(img_name)

        Images = loadmat(self.root + 'annotation/Images.mat')
        box_tmp = Images['Img'].squeeze()[image_id]['box'][0][0]
        skips = torch.ones((num_objs,), dtype=torch.int64)
        boxes = []
        for i in range(num_objs):
            tmp = list(map(int, box_tmp[i][0][0]))
            # tmp = list(box_tmp[i][0][0]) #[66, 135, 98, 251]
            if tmp[1] >= tmp[1] + tmp[3] or tmp[0] >= tmp[0] + tmp[2]:
                skips[i] = 0
                continue
            box = [tmp[1], tmp[0], tmp[1] + tmp[3], tmp[0] + tmp[2]]
            boxes.append(box)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3]) * (boxes[:, 2])

        labels = torch.ones((skips.sum(),), dtype=torch.int64)  # only person labels
        iscrowd = torch.zeros((skips.sum(),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([image_id2])
        # target['image_name'] = img_name
        target['num'] = torch.tensor([num_objs])
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)