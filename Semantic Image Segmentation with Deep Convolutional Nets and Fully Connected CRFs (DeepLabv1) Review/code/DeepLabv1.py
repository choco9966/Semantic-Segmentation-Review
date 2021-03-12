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
                                       conv_relu(1024, 1024, 1, 1), 
                                       nn.Dropout2d(0.5), 
                                       nn.Conv2d(1024, num_classes, 1)
                                       )
    def forward(self, x): 
        out = self.classifier(x)
        return out 

class DeepLabV1(nn.Module):
    ## VGG 위에 ASPP 쌓기
    def __init__(self, backbone, classifier, upsampling=8):
        super(DeepLabV1, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.upsampling = upsampling

    def forward(self, x):
        x = self.backbone(x)
        _, _, feature_map_h, feature_map_w = x.size()
        x = self.classifier(x)
        x = F.interpolate(x, size=(feature_map_h * self.upsampling, feature_map_w * self.upsampling), mode="bilinear")
        return x

if __name__=='__main__':
    backbone = VGG16()

    in_channels = 512
    out_channels = 256
    num_classes = 21

    classifier = classifier(num_classes=num_classes)

    model = DeepLabV1(backbone=backbone, classifier=classifier)
    #print(model)

    model.eval()
    image = torch.randn(1, 3, 1024, 512)
    print("input:", image.shape)
    print("output:", model(image).shape)
