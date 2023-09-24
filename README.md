# MSC-SSL
This repository is the official code for the paper "Cost-effective identification of the field maturity of tobacco leaves based on deep semi-supervised learning and smartphone photograph".

## Installation
Clone this repo.
```bash
git clone https://github.com/IFPI-LAB/MSC-SSL.git
cd MSC-SSL/
```

We have tested our code in the following environment:
- Python 3.8
- Pytorch 1.9.0
- Torchvision 0.10.0
- CUDA 11.1

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
To run the main Python file, use the following command:

```sh
python main.py --data_root your/data/path --model_save_path your/save/path --labeled_folder your/labeled/data
```
This command runs the MSC-SSL model with the specified hyperparameters. You can modify the command to run different experiments with different hyperparameters.

## Acknowledgement
This code is partially based on [DeepDA](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA), [Semi-Supervised-Transfer-Learning](https://github.com/SHI-Labs/Semi-Supervised-Transfer-Learning).