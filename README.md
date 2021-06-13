## Person Search Term Project

### Requirements

    - numpy>=1.18.1
    - opencv>=3.4.2
    - pytorch>=1.4.0
    - torchvision>=0.5
    - scikit-learn>=0.22.1

### Training

1. Download CUHK-SYSU dataset and unzip it in person_search/dataset.
2. Run ```data_preprocessing.py``` for pre-processing dataset
3. Run ```simclr_training.py``` for training SimCLR
4. Run ```person_detect.py``` for training Faster R-CNN for Person Detection
5. Run ```person_reID.py``` for computing similarities for Person Re-identification

### Source codes
1)Dataset folder tree

    person_search
     |
     +-- dataset
         |
         +-- CUHK-SYSU
             |
             +-- Image
             |   |
             |   +-- SSM
             |   +-- bbox (to be appeared by data_preprocessing.py)
             |       | 
             |       +-- TestG50
             |       +-- Train_only1
             +-- annotation
             |   | 
             |   +-- Images.mat
             |   +-- Person.mat
             |   +-- pool.mat
             |   +-- test
             |       | 
             |       +-- train_test
             |           | 
             |           +-- TestG50.mat
             +-- processed (to be appeared by data_preprocessing.py)
                 | 
                 +-- images.csv
                 +-- TestG50_data.csv	

2)Code tree

    person_search
     |
     +-- custom_data
     |   |
     |   +-- dataset.py
     |   +-- contrastive_learning.py
     |   +-- view_generator.py
     |   +-- gaussian_blur.py
     +-- models
     |   |
     |   +-- pretrained_model.py
     |   +-- resnet_simclr.py
     |   +-- simclr.py
     |
     +-- simclr_utils
     +-- torchvision_utils
     +-- vision
     +-- checkpoints (to be appeared during training step)
     +-- results (to be appeared during testing step)
     |
     +-- data_preprocessing.py
     +-- simclr_training.py
     +-- person_detect.py
     +-- person_reID.py
     +-- person_reID_only.py
