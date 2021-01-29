# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
class UNet(nn.Module):
    def __init__(self, num_classes=2):
        super(UNet, self).__init__()
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)
            return cbr

        # Contracting path 
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True)     
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
 
        self.enc3_1 = CBR2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0, bias=True)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, bias=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2)    

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0, bias=True)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0, bias=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2)    

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0, bias=True)
        self.enc5_2 = CBR2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0, bias=True)
        self.unpool4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=0, bias=True) 
        self.dec4_1 = CBR2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0, bias=True) 

        self.unpool3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0, bias=True) 
        self.dec3_1 = CBR2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, bias=True) 

        self.unpool2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True)  
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True)  

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True) 
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True) 
        self.score_fr = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0, bias=True) # Output Segmentation map 
        self._initialize_weights()

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)
        enc5_2 = self.enc5_2(enc5_1)

        unpool4 = self.unpool4(enc5_2)
        cat4 = torch.cat((unpool4, enc4_2[:,:,4:4+unpool4.size()[2],4:4+unpool4.size()[3]]), dim=1) 
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2[:,:,16:16+unpool3.size()[2],16:16+unpool3.size()[3]]), dim=1) 
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2[:,:,40:40+unpool2.size()[2],40:40+unpool2.size()[3]]), dim=1) 
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2[:,:,88:88+unpool1.size()[2],88:88+unpool1.size()[3]]), dim=1) 
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.score_fr(dec1_1) 
        return torch.sigmoid(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

'''
model = UNet(num_classes=2).cuda()
x = torch.randn([1, 1, 572 , 572]).cuda()
print("input shape : ", x.shape) # torch.Size([1, 1, 572, 572])
out = model(x)
print("output shape : ", out.size()) # torch.Size([1, 2, 388, 388])
'''