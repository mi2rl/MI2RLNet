import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
    
    
        def CBR2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels = input_channels, 
                                 out_channels = output_channels, 
                                 kernel_size = kernel_size,
                                 stride = stride,
                                 padding=padding, bias = bias)]
            layers += [nn.BatchNorm2d(num_features=output_channels)]
            layers += [nn.ReLU()]
            cbr = nn.Sequential(*layers)
            return cbr
        
        def start_block(input_channels, output_channels, kernel_size=2, stride=2, padding=0, bias=True):
            layers = []
        
            layers += [CBR2d(input_channels=input_channels, output_channels=output_channels)]
            layers += [CBR2d(input_channels=output_channels, output_channels=output_channels)]    
            
            start = nn.Sequential(*layers)
            return start
        
        def encoder_block(input_channels, output_channels, kernel_size=2, stride=2, padding=0, bias=True):
            layers = []
            
            layers += [nn.MaxPool2d(kernel_size=kernel_size)]
            layers += [CBR2d(input_channels=input_channels, output_channels=output_channels)]
            layers += [CBR2d(input_channels=output_channels, output_channels=output_channels)]
            
            
            enc = nn.Sequential(*layers)
            return enc
        
        def decoder_block(input_channels, output_channels, kernel_size=2, stride=2, padding=0, bias=True):
            layers = []
                        
           
            layers += [CBR2d(input_channels=input_channels*2, output_channels=output_channels)]
            layers += [CBR2d(input_channels=input_channels, output_channels=int(output_channels/2))]
            layers += [nn.ConvTranspose2d(in_channels=int(input_channels/2), out_channels=int(output_channels/2), 
                                          kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            
            dec = nn.Sequential(*layers)
            return dec
        
        def bridge(input_channels, output_channels, kernel_size=2, stride=2, padding=0, bias=True):
            layers = []
            layers += [nn.MaxPool2d(kernel_size=2)]
            layers += [CBR2d(input_channels=input_channels, output_channels=output_channels)]
            layers += [CBR2d(input_channels=output_channels, output_channels=input_channels)]

            bridge = nn.Sequential(*layers)
            return bridge
        
        def end_block(input_channels, output_channels, kernel_size=2, stride=2, padding=0, bias=True):
            layers = []
            layers += [CBR2d(input_channels=input_channels*2, output_channels=output_channels)]
            layers += [CBR2d(input_channels=output_channels, output_channels=input_channels)]
            
            end = nn.Sequential(*layers)
            return end
        
        self.encoder1 = start_block(3, 64)
        self.encoder2 = encoder_block(64, 128)
        self.encoder3 = encoder_block(128, 256)
        self.encoder4 = encoder_block(256, 512)
        
        self.bridge = bridge(512, 1024)
        self.upconv = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                         kernel_size=2, stride=2, padding=0, bias=True)
            
        self.decoder4 = decoder_block(512, 512)
        self.decoder3 = decoder_block(256, 256)
        self.decoder2 = decoder_block(128, 128)
        self.decoder1 = end_block(64, 64)
        
        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        

    def forward(self, x):
        encoder1 = self.encoder1(x)
        
        encoder2 = self.encoder2(encoder1)
        
        encoder3 = self.encoder3(encoder2)
        
        encoder4 = self.encoder4(encoder3)
        
        bridge = self.bridge(encoder4)
        
        upconv = self.upconv(bridge)
        
        
        cat4 = torch.cat((upconv, encoder4), dim=1)
        decoder4 = self.decoder4(cat4)
        
        cat3 = torch.cat((decoder4, encoder3), dim=1)
        decoder3 = self.decoder3(cat3)
        
        cat2 = torch.cat((decoder3, encoder2), dim=1)
        decoder2 = self.decoder2(cat2)
        
        cat1 = torch.cat((decoder2, encoder1), dim=1)
        decoder1 = self.decoder1(cat1)
        
        
        x = self.fc(decoder1)
        
        return x
