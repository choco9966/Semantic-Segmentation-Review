import torch
from torch import nn
import torch.nn.functional as F
from base_model import xception39, CBR


class SpatialPath(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialPath, self).__init__()
        hidden_channel = 64
        self.conv_7x7 = CBR(in_channels, hidden_channel, kernel_size=7, stride=2, padding=3)
        self.conv_3x3_1 = CBR(hidden_channel, hidden_channel, kernel_size=3, stride=2, padding=1)
        self.conv_3x3_2 = CBR(hidden_channel, hidden_channel, kernel_size=3, stride=2, padding=1)
        # conv_1x1는 왜 있지?
        self.conv_1x1 = CBR(hidden_channel, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        print("Spatial Path")
        x = self.conv_7x7(x)
        print(f"after SpatialPath layer1 shape : {x.shape}")
        x = self.conv_3x3_1(x)
        print(f"after SpatialPath layer2 shape : {x.shape}")
        x = self.conv_3x3_2(x)
        print(f"after SpatialPath layer3 shape : {x.shape}")
        x = self.conv_1x1(x)
        print(f"after SpatialPath 1x1 cov shape : {x.shape}")
        print("-"*20)
        return x

class AttentionRefinement(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionRefinement, self).__init__()
        # conv3x3 이거 왜있지
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # BN은 왜 없지,, 일단 넣음
        self.BN = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print("ARM) x.shape", x.shape)
        x = self.conv3x3(x)
        w = self.global_pool(x)
        print("ARM) after global pool", w.shape)
        w = self.conv1x1(w)
        print("ARM) after conv1x1", w.shape)
        w = self.BN(w)
        print("ARM) after BN", w.shape)
        w = nn.Sigmoid()(w)
        print("ARM) x, w", x.shape, w.shape)
        return x * w


class FeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels, reduction):
        super(FeatureFusion, self).__init__()
        mid_channels = out_channels // reduction

        self.cbr = CBR(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_1 = nn.Conv2d(out_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.activ = nn.ReLU(inplace=True)
        self.conv1x1_2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()


    def forward(self, fm1, fm2):
        x = torch.cat((fm1, fm2), dim=1)
        x = self.cbr(x)
        w = self.global_pool(x)
        w = self.conv1x1_1(w)
        w = self.activ(w)
        w = self.conv1x1_2(w)
        w = self.sigmoid(w)
        x_att = x * w
        x = x + x_att
        return x




class BiSeNet(nn.Module):
    def __init__(self, num_classes):
        super(BiSeNet, self).__init__()
        self.spatial_path = SpatialPath(3, 128)
        self.context_path = xception39()

        self.arms = nn.ModuleList([
            AttentionRefinement(256, 128),
            AttentionRefinement(128, 128)
            ])
        self.FFM = FeatureFusion(256, num_classes, 1)

    def forward(self, x):
        spatial_out = self.spatial_path(x)
        
        context_blocks = self.context_path(x)
        context_blocks.reverse()
        print("last layer output size of context_path", context_blocks[0].shape) # 1, 256, 16, 16 (32x down)

        # Global average pooling layer를 Xception 맨 뒤에 더함(receptive field를 최대로 키움)
        global_context = nn.AdaptiveAvgPool2d(1)(context_blocks[0]) 
        print("global_context after GlobalAveragePooling", global_context.shape) # 1, 256, 1, 1
        global_context = CBR(256, 128, kernel_size=1, stride=1, padding=0)(global_context)
        print("global_context after CBR 256->128", global_context.shape)
        global_context = F.interpolate(global_context,
                                        size=context_blocks[0].size()[2:],
                                        mode='bilinear', align_corners=True)
        print("global_context after interpolate", global_context.shape) # 1, 256, 16, 16
        last_fm = global_context
        
        for i, (fm, arm) in enumerate(zip(context_blocks[:2], self.arms)):
            print(f"[upsample {i}] feature_map shape", fm.shape) # 2, 256, 16, 16
            fm = arm(fm) # 256->128
            print(f"[upsample {i}] after ARM", fm.shape)
            fm += last_fm
            last_fm = F.interpolate(fm, size=(context_blocks[i+1].size()[2:]),
                        mode='bilinear', align_corners=True)
            print(f"[upsample {i}] after interpolate", fm.shape)
            # refine은 왜 하지? refine이 ConvBnRelu던데 (sparse한거 dense하게 만들어주려나)
            last_fm = CBR(128, 128, kernel_size=3, stride=1, padding=1)(last_fm) 
            print(f"[upsample {i}] after refine", fm.shape)
        
        context_out = last_fm

        concat_fm = self.FFM(spatial_out, context_out)
        upx8 = F.interpolate(
            concat_fm, scale_factor=8,
            mode='bilinear', align_corners=True)
        return upx8
        


if __name__ == "__main__":
    net = BiSeNet(num_classes=19)
    net.eval()
    input = torch.randn((2, 3, 512, 512)) # batch size=1이면 안됨
    out = net(input)
    print(out.shape)