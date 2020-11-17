import torch
import os

from .models.Unet import UNet

def build_MRA_BET(weight_path):
    model = UNet().cuda()
    weight = torch.load(weight_path)
    model.load_state_dict(weight['net'])
    
    return model

    
