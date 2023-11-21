import os
import numpy as np
import torch

from .wideresnetleaky import wideresnetleaky
from .resnet import  build_ResNet


def gen_model(depth, num_classes):
    orig_num_classes = num_classes

    model = build_ResNet(depth=depth,num_classes=orig_num_classes)

    return model