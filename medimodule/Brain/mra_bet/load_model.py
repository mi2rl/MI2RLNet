import torch
import os

from .model.Unet import UNet

def build_MRA_BET(weight_path, gpu_num):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    
    model = UNet().to(device)
    weight = torch.load(weight_path)#, map_location='cpu')
    model.load_state_dict(weight['net'])
    
    return model, device

    