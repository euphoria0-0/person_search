'''
Data preprocessing

Ref: None (for my own implementation)
'''
import pandas as pd
from imageio import imread, imsave
from scipy.io import loadmat
import os


def data_preprocessing(data_dir='dataset/CUHK-SYSU/'):
    os.makedirs(data_dir+'processed/', exist_ok=True)
    # 11814 persons
    imgs = loadmat(data_dir+'annotation/Images.mat')
    imgs = pd.DataFrame(imgs['Img'].reshape(-1))
    print(imgs.shape)
    imgs['imname'] = [x[0] for x in imgs['imname']]
    imgs['nAppear'] = [x[0][0] for x in imgs['nAppear']]
    imgs['box'] = [x.squeeze() for x in imgs['box']]
    imgs.tail(2)
    imgs.to_csv(data_dir+'processed/images.csv', index=False)

    os.makedirs(data_dir + 'Image/bbox/', exist_ok=True)
    os.makedirs(data_dir + 'Image/bbox/Train_only1/', exist_ok=True)
    os.makedirs(data_dir + 'Image/bbox/TestG50/', exist_ok=True)
    # persons for training (but only one image of each person)
    train = loadmat(data_dir+'/annotation/test/train_test/Train.mat')
    train = train['Train']
    for img in train:
        idname = img[0][0][0][0][0]
        nAppear = img[0][0][0][1][0][0]
        scene = {}
        for i,x in enumerate(img[0][0][0][2][0][:1]):
            scene[i] = {
                'imname': x[0][0],
                'idlocate': list(x[1][0]),
                'ishard': x[2][0][0]
            }
            image = imread(data_dir+'Image/SSM/'+scene[i]['imname'])
            box = list(map(int, scene[i]['idlocate']))
            crop = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
            os.makedirs(data_dir+'Image/bbox/Train_only1/'+idname, exist_ok=True)
            imsave(data_dir+'Image/bbox/Train_only1/'+idname+'/'+idname+'_'+scene[i]['imname'], crop)


    # 2900 query persons with gallery size(50) for testing
    data_dir = 'dataset/CUHK-SYSU/'
    test = loadmat(data_dir+'annotation/test/train_test/TestG50.mat')
    test = test['TestG50'].squeeze()
    for i in range(test.shape[0]):
        x = test[i][0][0][0]
        imname = x[0][0]
        idlocate = x[1][0].tolist()
        ishard = x[2][0][0]
        idname = x[3][0]

        image = imread(data_dir+'Image/SSM/'+imname)
        box = list(map(int, idlocate))
        crop = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        os.makedirs(data_dir+'Image/bbox/TestG50/'+idname, exist_ok=True)
        imsave(data_dir+'Image/bbox/TestG50/'+idname+'/'+idname+'_'+imname, crop)
