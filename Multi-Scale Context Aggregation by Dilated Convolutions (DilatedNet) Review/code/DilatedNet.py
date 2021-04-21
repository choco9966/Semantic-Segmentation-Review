import torch
import torch.nn as nn
from torch.nn import functional as F

def conv_relu(in_ch, out_ch, size=3, rate=1):
    conv_relu = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=size, 
                                    stride=1, padding=rate, dilation=rate),
                                 nn.ReLU())
    return conv_relu


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features1 = nn.Sequential(conv_relu(3, 64, 3, 1),
                                      conv_relu(64, 64, 3, 1),
                                      nn.MaxPool2d(3, stride=2, padding=1))
        self.features2 = nn.Sequential(conv_relu(64, 128, 3, 1),
                                      conv_relu(128, 128, 3, 1),
                                      nn.MaxPool2d(3, stride=2, padding=1))
        self.features3 = nn.Sequential(conv_relu(128, 256, 3, 1),
                                      conv_relu(256, 256, 3, 1),
                                      conv_relu(256, 256, 3, 1),
                                      nn.MaxPool2d(3, stride=2, padding=1))
        self.features4 = nn.Sequential(conv_relu(256, 512, 3, 1),
                                      conv_relu(512, 512, 3, 1),
                                      conv_relu(512, 512, 3, 1),
                                      nn.MaxPool2d(3, stride=1, padding=1))
                                      # and replace subsequent conv layer r=2
        self.features5 = nn.Sequential(conv_relu(512, 512, 3, rate=2),
                                      conv_relu(512, 512, 3, rate=2),
                                      conv_relu(512, 512, 3, rate=2),
                                      nn.MaxPool2d(3, stride=1, padding=1), 
                                      nn.AvgPool2d(3, stride=1, padding=1)) # 마지막 stride=1로 해서 두 layer 크기 유지 
    def forward(self, x):
        out = self.features1(x)
        out = self.features2(out)
        out = self.features3(out)
        out = self.features4(out)
        out = self.features5(out)
        return out

class classifier(nn.Module):
    def __init__(self, num_classes): 
        super(classifier, self).__init__()
        self.classifier = nn.Sequential(conv_relu(512, 1024, 3, rate=12), 
                                       nn.Dropout2d(0.5), 
                                       conv_relu(1024, 1024, 1, rate=0), 
                                       nn.Dropout2d(0.5), 
                                       nn.Conv2d(1024, num_classes, kernel_size=1)
                                       )
    def forward(self, x): 
        out = self.classifier(x)
        return out 

class BasicContextModule(nn.Module):
    def __init__(self, num_classes):
        super(BasicContextModule, self).__init__()
        
        self.layer1 = nn.Sequential(conv_relu(num_classes, num_classes, 3, 1))
        self.layer2 = nn.Sequential(conv_relu(num_classes, num_classes, 3, 1))
        self.layer3 = nn.Sequential(conv_relu(num_classes, num_classes, 3, 2))
        self.layer4 = nn.Sequential(conv_relu(num_classes, num_classes, 3, 4))
        self.layer5 = nn.Sequential(conv_relu(num_classes, num_classes, 3, 8))
        self.layer6 = nn.Sequential(conv_relu(num_classes, num_classes, 3, 16))
        self.layer7 = nn.Sequential(conv_relu(num_classes, num_classes, 3, 1))
        # No Truncation 
        self.layer8 = nn.Sequential(nn.Conv2d(num_classes, num_classes, 1, 1))
        
    def forward(self, x): 
        
        out = self.layer1(x)
        out = self.layer2(x)
        out = self.layer3(x)
        out = self.layer4(x)
        out = self.layer5(x)
        out = self.layer6(x)
        out = self.layer7(x)
        out = self.layer8(x)
        
        return out
    
class DilatedNet(nn.Module):
    def __init__(self, backbone, classifier, context_module):
        super(DilatedNet, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.context_module = context_module
        
        self.deconv = nn.ConvTranspose2d(in_channels=12,
                                         out_channels=12,
                                         kernel_size=16,
                                         stride=8,
                                         padding=4)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        x = self.context_module(x)
        x = self.deconv(x)
        return x
