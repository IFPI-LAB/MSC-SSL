# MSC-SSL
This repository is the official code for the paper "Cost-effective identification of the field maturity of tobacco leaves based on deep semi-supervised learning and smartphone photograph".

## Installation
Clone this repo.
```bash
git clone https://github.com/IFPI-LAB/MSC-SSL.git
cd MSC-SSL/
```

We have tested our code in the following environment:
- Python 3.7
- Pytorch 1.10.1
- Torchvision 0.10.1
- CUDA 11.3

## Data Preparation
The dataset is not publicly available, you may contact the corresponding author for research purposes to obtain the dataset.

You can organize your dataset into directories, as shown below:
```
- data_root
    - labeled
        - labeled_folder
            - class_1
            - class_2
            - ...
    - train
        - class_1
        - class_2
        - ...
    - test
```

## Running the model

### Baseline methods
Download the pretrained model [checkpoint](https://drive.google.com/file/d/1VClK73Dgu0RRKw9Cnhb0NjytGJLDAyLN/view?usp=sharing) and put it under `./ckpt`. Then, run the following command

```sh
python main.py --data_root your/data/path --model_save_path your/save/path --labeled_folder your/labeled/data
```
This command runs the MSC-SSL model with the specified hyperparameters. You can modify the command to run different experiments with different hyperparameters.

## Sample Selection
### Self-Supervised Training
To train a self-supervised network based on [SimSiam](https://github.com/facebookresearch/simsiam) using the unlabeled image samples, you can run the following command to train from scratch:
```sh
cd simsiam
python train_simsiam.py --data your/data/path --save_path your/save/path --lars
```
Or you can use the following command to train the network using a pre-trained checkpoint:
```sh
python train_simsiam.py --pretrained pretrained/checkpoint/path --data your/data/path --save_path your/save/path --lars
```

### Feature Extraction
Then use the model obtained from training above to extract features from the unlabelled data:
```sh
python feature_extract.py --model_path your/model/path --data_path your/data/path --save_npz your/save/path
```

### Sample Selection


## Acknowledgement
This code is partially based on [DeepDA](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA), [Semi-Supervised-Transfer-Learning](https://github.com/SHI-Labs/Semi-Supervised-Transfer-Learning).