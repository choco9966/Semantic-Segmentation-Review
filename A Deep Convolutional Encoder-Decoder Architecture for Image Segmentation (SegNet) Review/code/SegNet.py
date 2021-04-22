import torch
import torch.nn as nn
from torchvision import models



class SegNet(nn.Module):
    def __init__(self, num_classes=21, init_weights=True):
        super(SegNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        
        def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, 
                          out_channels=out_channels,
                          kernel_size=kernel_size, 
                          stride=stride, 
                          padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())

        
        '''
        input: "data"
        input_dim: 1
        input_dim: 3
        input_dim: 224
        input_dim: 224
        '''
        
        # 224 x 224
        # conv1
        self.conv1_1 = CBR(3, 64, 3, 1, 1)
        self.conv1_2 = CBR(64, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True) # 1/2
        
        # 112 x 112
        # conv2 
        self.conv2_1 = CBR(64, 128, 3, 1, 1)
        self.conv2_2 = CBR(128, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True) # 1/4
        
        # 56 x 56
        # conv3
        self.conv3_1 = CBR(128, 256, 3, 1, 1)
        self.conv3_2 = CBR(256, 256, 3, 1, 1)
        self.conv3_3 = CBR(256, 256, 3, 1, 1)        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True) # 1/8
        
        # 28 x 28
        # conv4
        self.conv4_1 = CBR(256, 512, 3, 1, 1)
        self.conv4_2 = CBR(512, 512, 3, 1, 1)
        self.conv4_3 = CBR(512, 512, 3, 1, 1)        
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True) # 1/16
        
        # 14 x 14
        # conv5
        self.conv5_1 = CBR(512, 512, 3, 1, 1)
        self.conv5_2 = CBR(512, 512, 3, 1, 1)
        self.conv5_3 = CBR(512, 512, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True)
        
        # 14 x 14
        # unpool5
        self.unpool5 = nn.MaxUnpool2d(2, stride=2)
        self.deconv5_1 = CBR(512, 512, 3, 1, 1)
        self.deconv5_2 = CBR(512, 512, 3, 1, 1)
        self.deconv5_3 = CBR(512, 512, 3, 1, 1)
        
        # 28 x 28
        # unpool4
        self.unpool4 = nn.MaxUnpool2d(2, stride=2)
        self.deconv4_1 = CBR(512, 512, 3, 1, 1)
        self.deconv4_2 = CBR(512, 512, 3, 1, 1)
        self.deconv4_3 = CBR(512, 256, 3, 1, 1)        

        # 56 x 56        
        # unpool3
        self.unpool3 = nn.MaxUnpool2d(2, stride=2)
        self.deconv3_1 = CBR(256, 256, 3, 1, 1)
        self.deconv3_2 = CBR(256, 256, 3, 1, 1)
        self.deconv3_3 = CBR(256, 128, 3, 1, 1)                          
        
        # 112 x 112         
        # unpool2
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.deconv2_1 = CBR(128, 128, 3, 1, 1)
        self.deconv2_2 = CBR(128, 64, 3, 1, 1)

        # 224 x 224        
        # unpool1
        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.deconv1_1 = CBR(64, 64, 3, 1, 1)
        # Score
        self.score_fr = nn.Conv2d(64, num_classes, 3, 1, 1, 1)
        
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        
        h = self.conv1_1(x)
        h = self.conv1_2(h)
        h, pool1_indices = self.pool1(h)
        
        h = self.conv2_1(h)
        h = self.conv2_2(h)
        h, pool2_indices = self.pool2(h)
        
        h = self.conv3_1(h)
        h = self.conv3_2(h)
        h = self.conv3_3(h)        
        h, pool3_indices = self.pool3(h)
        
        h = self.conv4_1(h)
        h = self.conv4_2(h)
        h = self.conv4_3(h)        
        h, pool4_indices = self.pool4(h) 
        
        h = self.conv5_1(h)
        h = self.conv5_2(h)
        h = self.conv5_3(h)        
        h, pool5_indices = self.pool5(h)
        
        
        h = self.unpool5(h, pool5_indices)
        h = self.deconv5_1(h)        
        h = self.deconv5_2(h)                
        h = self.deconv5_3(h)                

        h = self.unpool4(h, pool4_indices)
        h = self.deconv4_1(h)        
        h = self.deconv4_2(h)                
        h = self.deconv4_3(h)                       

        h = self.unpool3(h, pool3_indices)
        h = self.deconv3_1(h)        
        h = self.deconv3_2(h)                
        h = self.deconv3_3(h)                            
        
        h = self.unpool2(h, pool2_indices)
        h = self.deconv2_1(h)        
        h = self.deconv2_2(h)                                         

        h = self.unpool1(h, pool1_indices)
        h = self.deconv1_1(h)                                        
            
        h = self.score_fr(h)           
        
        return h
