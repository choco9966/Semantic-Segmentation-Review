import torch
import torch.nn as nn
from torch.nn import functional as F

def conv3x3_relu(in_ch, out_ch, rate=1):
    conv3x3_relu = nn.Sequential(nn.Conv2d(in_ch, 
                                           out_ch, 
                                           kernel_size=3, 
                                           stride=1, 
                                           padding=rate, 
                                           dilation=rate),
                                 nn.ReLU())
    return conv3x3_relu

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(conv3x3_relu(3, 64),
                                      conv3x3_relu(64, 64),
                                      nn.MaxPool2d(3, stride=2, padding=1),
                                      conv3x3_relu(64, 128),
                                      conv3x3_relu(128, 128),
                                      nn.MaxPool2d(3, stride=2, padding=1),
                                      conv3x3_relu(128, 256),
                                      conv3x3_relu(256, 256),
                                      conv3x3_relu(256, 256),
                                      nn.MaxPool2d(3, stride=2, padding=1),
                                      conv3x3_relu(256, 512),
                                      conv3x3_relu(512, 512),
                                      conv3x3_relu(512, 512),
                                      nn.MaxPool2d(3, stride=1, padding=1), # 마지막 stride=1로 해서 두 layer 크기 유지 
                                      # and replace subsequent conv layer r=2
                                      conv3x3_relu(512, 512, rate=2),
                                      conv3x3_relu(512, 512, rate=2),
                                      conv3x3_relu(512, 512, rate=2),
                                      nn.MaxPool2d(3, stride=1, padding=1)) # 마지막 stride=1로 해서 두 layer 크기 유지 
    
        
    def forward(self, x):
        output = self.features(x)
        return output
      
class ASPPConv(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, padding, dilation):
        super(ASPPConv, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        output = self.relu(x)
        return output

class ASPPPooling(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(ASPPPooling, self).__init__()
        self.globalavgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(inplanes, outplanes, 1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.globalavgpool(x)
        x = self.conv(x)
        x = self.bn(x)
        output = self.relu(x)
        return output


class ASPP(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]
        self.aspp1 = ASPPConv(inplanes, outplanes, 1, padding=0, dilation=dilations[0])
        self.aspp2 = ASPPConv(inplanes, outplanes, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = ASPPConv(inplanes, outplanes, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = ASPPConv(inplanes, outplanes, 3, padding=dilations[3], dilation=dilations[3])
        self.global_avg_pool = ASPPPooling(inplanes, outplanes)
        self.project = nn.Sequential(
            nn.Conv2d(outplanes*5, outplanes, 1, bias=False), 
            nn.BatchNorm2d(outplanes), 
            nn.ReLU(), 
            nn.Dropout(0.5)      
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        output = self.project(x)
        return output
     
class DeepLabHead(nn.Sequential):
    def __init__(self, in_ch, out_ch, n_classes):
        super(DeepLabHead, self).__init__()
        self.add_module("0", ASPP(in_ch, out_ch))
        self.add_module("1", nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1 , bias=False))
        self.add_module("2", nn.BatchNorm2d(out_ch))
        self.add_module("3", nn.ReLU())
        self.add_module("4", nn.Conv2d(out_ch, n_classes, kernel_size=1, stride=1))
        
        
class DeepLabV3(nn.Sequential):
    def __init__(self, n_classes, n_blocks, atrous_rates):
        super(DeepLabV3, self).__init__()
        self.backbone = VGG16()
        self.classifier = DeepLabHead(in_ch=512, out_ch=256, n_classes=12)

    def forward(self, x): 
        h = self.backbone(x)
        h = self.classifier(h)
        output = F.interpolate(h, size=x.shape[2:], mode="bilinear", align_corners=False)
        return output
      
if __name__ == '__main__': 
  # 구현된 model에 임의의 input을 넣어 output이 잘 나오는지 test
  model = DeepLabV3(n_classes=12, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24])

  x = torch.randn([2, 3, 512, 512])
  print("input shape : ", x.shape)
  out = model(x).to(device)
  print("output shape : ", out.size())

  model = model.to(device)
