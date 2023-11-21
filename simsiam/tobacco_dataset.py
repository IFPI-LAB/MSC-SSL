import os
import glob
import csv
import random

from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class TBCdataset(Dataset):
    def __init__(self, root):
        super(TBCdataset, self).__init__()
        self.root = root
        self.images1 = os.listdir(os.path.join(root, '1-unripe'))
        self.images2 = os.listdir(os.path.join(root, '2-ripe'))
        self.images3 = os.listdir(os.path.join(root, '3-overripe'))

        self.labels1 = torch.zeros(len(self.images1), 1)
        self.labels2 = torch.ones(len(self.images2), 1)
        self.labels3 = torch.ones(len(self.images3), 1) * 2

        self.images = self.images1 + self.images2 + self.images3
        self.labels = torch.cat([self.labels1, self.labels2, self.labels3], dim=0)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.labels[idx]
        im_name = self.images[idx]
        if label == 0:
            im_path = os.path.join(self.root, '1-unripe', im_name)
        elif label == 1:
            im_path = os.path.join(self.root, '2-ripe', im_name)
        elif label == 2:
            im_path = os.path.join(self.root, '3-overripe', im_name)

        img = Image.open(im_path).convert('RGB')
        img = self.transform(img)

        return img, label, im_path