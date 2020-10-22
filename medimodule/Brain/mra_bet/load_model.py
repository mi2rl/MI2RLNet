import torch
import os

from .models.Unet import UNet

def build_MRA_BET(weight_path, gpu_num):
    model = UNet().cuda()
    weight = torch.load(weight_path)#, map_location='cpu')
    model.load_state_dict(weight['net'])
    
    return model, device

    
