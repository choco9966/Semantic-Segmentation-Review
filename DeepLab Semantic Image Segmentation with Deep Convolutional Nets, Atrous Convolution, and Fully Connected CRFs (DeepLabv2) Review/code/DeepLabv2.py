import torch
import torch.nn as nn
from torch.nn import functional as F
"""
last few max pooling layers를 없애고, 대신에 이후 convolution layers에 atrous convolution을 함. 
VGG16 or ResNet101 in fully convolutional fashion + using Atrous conv for downsampling
bilinear interpolation to original resolution
vgg16 based ASPP-L (4 branches : 6, 12, 18, 24)
"""

def conv3x3_relu(in_ch, out_ch, rate=1):
    conv3x3_relu = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, 
                                    stride=1, padding=rate, dilation=rate),
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
        return self.features(x)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=1024, num_classes=21):
        super(ASPP, self).__init__()
        # atrous 3x3, rate=6
        self.conv_3x3_r6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        # atrous 3x3, rate=12
        self.conv_3x3_r12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        # atrous 3x3, rate=18
        self.conv_3x3_r18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        # atrous 3x3, rate=24
        self.conv_3x3_r24 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=24, dilation=24)
        self.bn_conv_3x3 = nn.BatchNorm2d(out_channels)

        self.conv_1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1 = nn.BatchNorm2d(out_channels)

        self.conv_1x1_out = nn.Conv2d(out_channels, num_classes, kernel_size=1)
        self.bn_conv_1x1_out = nn.BatchNorm2d(num_classes)

    def forward(self, feature_map):
        # 1번 branch
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_r6 = F.relu(self.bn_conv_3x3(self.conv_3x3_r6(feature_map)))
        out_img_r6 = F.relu(self.bn_conv_1x1(self.conv_1x1(out_3x3_r6)))
        out_img_r6 = F.relu(self.bn_conv_1x1_out(self.conv_1x1_out(out_img_r6)))
        # 2번 branch
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_r12 = F.relu(self.bn_conv_3x3(self.conv_3x3_r12(feature_map)))
        out_img_r12 = F.relu(self.bn_conv_1x1(self.conv_1x1(out_3x3_r12)))
        out_img_r12 = F.relu(self.bn_conv_1x1_out(self.conv_1x1_out(out_img_r12)))
        # 3번 branch
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_r18 = F.relu(self.bn_conv_3x3(self.conv_3x3_r18(feature_map)))
        out_img_r18 = F.relu(self.bn_conv_1x1(self.conv_1x1(out_3x3_r18)))
        out_img_r18 = F.relu(self.bn_conv_1x1_out(self.conv_1x1_out(out_img_r18)))
        # 4번 branch
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_r24 = F.relu(self.bn_conv_3x3(self.conv_3x3_r24(feature_map)))
        out_img_r24 = F.relu(self.bn_conv_1x1(self.conv_1x1(out_3x3_r24)))
        out_img_r24 = F.relu(self.bn_conv_1x1_out(self.conv_1x1_out(out_img_r24)))

        out = sum([out_img_r6, out_img_r12, out_img_r18, out_img_r24])
        
        return out

class DeepLabV2(nn.Module):
    ## VGG 위에 ASPP 쌓기
    def __init__(self, backbone, classifier, upsampling=8):
        super(DeepLabV2, self).__init__()
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
    aspp_module = ASPP(in_channels, out_channels, num_classes)

    model = DeepLabV2(backbone=backbone, classifier=aspp_module)
    #print(model)

    model.eval()
    image = torch.randn(1, 3, 1024, 512)
    print("input:", image.shape)
    print("output:", model(image).shape)
