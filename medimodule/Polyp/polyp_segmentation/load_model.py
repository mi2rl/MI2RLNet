import torch
import torch.nn as nn
from .model.UNet import UNet
    
def build_polyp_segmentation(weight_path = './weight.pth'):
    model = UNet()
    
    weight = torch.load(weight_path, map_location='cpu')
    
    model.load_state_dict(weight['net'])
    
    return model