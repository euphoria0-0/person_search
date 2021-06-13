# Person Search Term Project

### Requirements

    - numpy>=1.18.1
    - opencv>=3.4.2
    - pytorch>=1.4.0
    - torchvision>=0.5
    - scikit-learn>=0.22.1

### Training

1. Download CUHK-SYSU dataset and unzip it in person_search/dataset.
2. Run ```simclr_training.py``` for representation learning.
3. Run ```PersonSearch.py``` for person search.

### Source codes
1) Dataset folder tree

```
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
         |
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
         |
         +-- processed (to be appeared by data_preprocessing.py)
             | 
             +-- images.csv
             +-- TestG50_data.csv	
```

2) Code tree

```
person_search
 |
 +-- custom_data
 |   |
 |   +-- dataset.py (our own implementation)
 |   +-- data_preprocessing.py (our own implementation)
 |   +-- contrastive_learning.py
 |   +-- view_generator.py
 |   +-- gaussian_blur.py
 |
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
 +-- simclr_training.py
 +-- PersonSearch.py
```
