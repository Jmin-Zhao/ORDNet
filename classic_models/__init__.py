import random
import os
import numpy as np
import torch



from .resnet import  resnet34, resnet50, resnet101, resnext50_32x4d, resnext101_32x8d
from .mobilenet_v3 import mobilenet_v3_small, mobilenet_v3_large
from .shufflenet_v2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0
from .squeezenet import squeezenet1_1

cfgs = {
    'resnet_small': resnet34,
    'resnet': resnet50,
    'resnet_big': resnet101,
    'resnext': resnext50_32x4d,
    'resnext_big': resnext101_32x8d,
    'mobilenet_v3': mobilenet_v3_small,
    'mobilenet_v3_large': mobilenet_v3_large,
    'shufflenet_small':shufflenet_v2_x0_5,
    'shufflenet': shufflenet_v2_x1_0,
    'squeezenet_v1_1': squeezenet1_1,
}

def find_model_using_name(model_name, num_classes):   
    return cfgs[model_name](num_classes)

 
