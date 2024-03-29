# -*- coding: utf-8 -*-
"""ENet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13X0E5Y6DLsDUETp_1Nkjbiz7Ib0FD3yX
"""

import torch.nn as nn 
import torch 
# from modules import InitialBlock, Bottleneck, DownsamplingBottleneck, UpsamplingBottleneck

class InitialBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, bias=False): 
        super(InitialBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels-3, 
                              kernel_size=3, stride=2, padding=1, bias=bias)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x): 
        conv = self.conv(x) 
        maxpool = self.maxpool(x) 
        out = torch.cat((conv, maxpool), axis=1)
        out = self.bn(out) 
        out = self.prelu(out) 
        return out

class Bottleneck(nn.Module): 
    # SyntaxError: non-default argument follows default argument
    # Non-default가 먼저 와야함 
    def __init__(self, in_channels, out_channels, prob, mode, dilation, bias=False): 
        super(Bottleneck, self).__init__()
        mid_channels = out_channels // 4
        if mode == 'dilation': 
            self.ext_branch = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, 
                          stride=1, bias=bias), 
                nn.BatchNorm2d(mid_channels), 
                nn.PReLU(), 

                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                          stride=1, bias=bias, dilation=dilation, padding=dilation), 
                nn.BatchNorm2d(mid_channels), 
                nn.PReLU(),   

                nn.Conv2d(mid_channels, out_channels, kernel_size=1, 
                          stride=1, bias=bias), 
                nn.BatchNorm2d(out_channels), 
                nn.PReLU(), 
                nn.Dropout2d(p=prob) 
            )

        if mode == 'asymmetric': 
            self.ext_branch = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, 
                          stride=1, bias=bias), 
                nn.BatchNorm2d(mid_channels), 
                nn.PReLU(), 

                # out_cxhxwinput_c
                nn.Conv2d(mid_channels, mid_channels, kernel_size=[5, 1], 
                          stride=1, bias=bias, padding=(2,0)), 
                nn.BatchNorm2d(mid_channels), 
                nn.PReLU(),    
                nn.Conv2d(mid_channels, mid_channels, kernel_size=[1, 5], 
                          stride=1, bias=bias, padding=(0,2)), 
                nn.BatchNorm2d(mid_channels), 
                nn.PReLU(),  

                nn.Conv2d(mid_channels, out_channels, kernel_size=1, 
                          stride=1, bias=bias), 
                nn.BatchNorm2d(out_channels), 
                nn.PReLU(), 
                nn.Dropout2d(p=prob) 
            )      
        if mode == 'None' or mode == 'none': 
            self.ext_branch = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, 
                          stride=1, bias=bias), 
                nn.BatchNorm2d(mid_channels), 
                nn.PReLU(), 

                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                          stride=1, bias=bias, padding=1), 
                nn.BatchNorm2d(mid_channels), 
                nn.PReLU(), 

                nn.Conv2d(mid_channels, out_channels, kernel_size=1, 
                          stride=1, bias=bias), 
                nn.BatchNorm2d(out_channels), 
                nn.PReLU(), 
                nn.Dropout2d(p=prob) 
            )
        self.prelu = nn.PReLU()

    def forward(self, x): 
        main = x
        out = self.ext_branch(x) 
        out = out + main
        out = self.prelu(out)
        return out

import torch.nn.functional as F
class DownsamplingBottleneck(nn.Module): 
    def __init__(self, in_channels, out_channels, prob, bias=False): 
        super(DownsamplingBottleneck, self).__init__()
        self.DownSample = True
        mid_channels = out_channels // 4
        self.ext_branch = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=2, stride=2, bias=bias), 
                nn.BatchNorm2d(mid_channels), 
                nn.PReLU(), 
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
                nn.BatchNorm2d(mid_channels), 
                nn.PReLU(), 
                nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=bias), 
                nn.BatchNorm2d(out_channels), 
                nn.PReLU(), 
                nn.Dropout2d(p=prob) #  이게 sptial dropout은 아닌 것 같은데 코드가 없는 것 같음 
            )
            
        self.main_branch = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True), 
        )
        self.PReLU = nn.PReLU()

    def forward(self, x): 
        ext_branch = self.ext_branch(x) 
        main_branch, pool_indices = self.main_branch(x)
        dc = ext_branch.size()[1] - main_branch.size()[1]
        bs, h, w = x.shape[0], ext_branch.size()[2], ext_branch.size()[3]

        pad = torch.zeros([bs, dc, h, w])
        main_branch = torch.cat((main_branch, pad), axis=1)
        out = main_branch + ext_branch 
        out = self.PReLU(out) 
        return out, pool_indices

import torch.nn.functional as F
class UpsamplingBottleneck(nn.Module): 
    def __init__(self, in_channels, out_channels, prob, bias=False): 
        super(UpsamplingBottleneck, self).__init__()
        self.Upsample = True 
        mid_channels = out_channels // 4
        self.ext_branch = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=bias),
                nn.BatchNorm2d(mid_channels), 
                nn.PReLU(), 

                nn.ConvTranspose2d(mid_channels, mid_channels, 
                                kernel_size=2, stride=2, bias=bias), 
                nn.BatchNorm2d(mid_channels), 
                nn.PReLU(), 
                
                nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=bias),
                nn.BatchNorm2d(out_channels), 
                nn.PReLU(), 
                nn.Dropout2d(p=prob)
            )
            
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels), 
        )
        self.main_unpool = nn.MaxUnpool2d(kernel_size=2)
        self.PReLU = nn.PReLU()

    def forward(self, x, max_indices): 
        ext_branch = self.ext_branch(x) 
        main_branch = self.main_branch(x) 
        main_branch = self.main_unpool(main_branch, max_indices) 

        out = ext_branch + main_branch 
        out = self.PReLU(out)
        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.initial_block = InitialBlock(3, 16, bias=False)
        self.layers = nn.ModuleList()
        self.layers += [DownsamplingBottleneck(16,64,prob=0.1)]
        for _ in range(4): 
            self.layers += [Bottleneck(in_channels=64, out_channels=64, bias=False, prob=0.1, mode='None',dilation=1)]
        # bottleneck 2.0-8
        # bottleneck 3.1-8
        self.layers += [DownsamplingBottleneck(64,128,prob=0.1)]
        for _ in range(2): 
            self.layers += [Bottleneck(128,128, prob=0.1, mode='None', dilation=1)]
            self.layers += [Bottleneck(128,128, prob=0.1, mode='dilation', dilation=2)]
            self.layers += [Bottleneck(128,128, prob=0.1, mode='asymmetric', dilation=1)]
            self.layers += [Bottleneck(128,128, prob=0.1, mode='dilation', dilation=4)]
            self.layers += [Bottleneck(128,128, prob=0.1, mode='None', dilation=1)]
            self.layers += [Bottleneck(128,128, prob=0.1, mode='dilation', dilation=8)]
            self.layers += [Bottleneck(128,128, prob=0.1, mode='asymmetric', dilation=1)]
            self.layers += [Bottleneck(128,128, prob=0.1, mode='dilation', dilation=16)]

    def forward(self, x): 
        out = self.initial_block(x)
        pooling_stack = []
        for layer in self.layers:
            if hasattr(layer, 'DownSample'):
                out, pooling_indices = layer(out)
                pooling_stack.append(pooling_indices)
            else:
                out = layer(out)      
        return out, pooling_stack

class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder,self).__init__()
        # bottleneck 4.0-2
        self.layers = nn.ModuleList()
        self.layers += [UpsamplingBottleneck(128, 64, prob=0.1)]
        self.layers += [Bottleneck(64,64, prob=0.1, mode='None', dilation=1)]
        self.layers += [Bottleneck(64,64, prob=0.1, mode='None', dilation=1)]

        # bottleneck 5.0-2
        self.layers += [UpsamplingBottleneck(64, 16, prob=0.1)]
        self.layers += [Bottleneck(16,16, prob=0.1, mode='None', dilation=1)]

        # fullconv 
        self.layers += [nn.ConvTranspose2d(16, num_classes, kernel_size=2, 
                                           stride=2, padding=0, bias=False)]

    def forward(self, x, pooling_stack):
        out = x
        for layer in self.layers:
            if hasattr(layer, 'Upsample'):
                pooling_indices = pooling_stack.pop()
                out = layer(out, pooling_indices)
            else:
                out = layer(out)
        return out

class ENet(nn.Module):
    def __init__(self, num_classes):
        super(ENet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_classes=num_classes)

    def forward(self, x):
        out, pooling_stack = self.encoder(x)
        out = self.decoder(out, pooling_stack)
        return out

ENet = ENet(num_classes=12)

ENet.eval() 
input = torch.rand(2, 3, 512, 512)
output = ENet(input)

print(output.size())