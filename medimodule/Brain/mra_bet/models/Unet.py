import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
          nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1),
          nn.BatchNorm2d(mid_channels),
          nn.ReLU(inplace=True),
          nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels,out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels,out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels//2)
        else: 
            self.up = nn.ConvTranspose2d(in_channels,in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels,out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2,x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self,x):
        return self.conv(x)
    

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, dims=64, bilinear=True):
        super(UNet,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.In = DoubleConv(n_channels,dims)
        self.down1 = Down(dims,dims*2)
        self.down2 = Down(dims*2,dims*4)
        self.down3 = Down(dims*4,dims*8)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(dims*8,dims*16//factor)
        
        self.up1 = Up(dims*16,dims*8//factor, bilinear)
        self.up2 = Up(dims*8,dims*4//factor, bilinear)
        self.up3 = Up(dims*4,dims*2//factor, bilinear)
        self.up4 = Up(dims*2,dims, bilinear)
        self.Out = OutConv(dims, n_channels)
           
    def forward(self, x):
        x1 = self.In(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        output = self.Out(x)
        return output