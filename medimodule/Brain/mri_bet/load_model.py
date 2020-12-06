import torch
import os

from .models.Unet import UNet

def build_MRI_BET(weight_path):
    model = UNet().cuda()
    weight = torch.load(weight_path, map_location='cuda:0')
    model.load_state_dict(weight['net'])
    
    return model

    
