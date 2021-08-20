# code implementation

**Check Point**

- [ ]  input image에 대하여 scale 기법은 어떤 기법을 사용하는가? (max-pooling, average-pooling, **bilinear**)
- [ ]  Backbone에서의 nn.AvgPool2d 제거 (DeepLabLargeFOV에서는 존재)
- [ ]  Attention to Scale 부분에서 scale에 따라 입력되는 input을 `concat`하므로 channel이 총 $1024 \times s$ 로 결정되며, convolution을 통해 dimension reduction이 발생 ($1024 \times s$ →  $512$ → 2 )
- [ ]  Attention to Scale 의 Weights 를 slicing 하는 방법이 적절한지 (e.g.  $s = 2$ 로 설명)
- [ ]  `weight  $\times$ Score map I`
- [ ]  Extra-Supervision for loss

### Backbone (DeepLabLargeFOV)

---

- **conv1 ~ FC7**

![Untitled](code%20implementation%20ac14e91201d34deea2b5cbc8abe0f82b/Untitled.png)

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

print('pytorch version: {}'.format(torch.__version__))
print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

print(torch.cuda.get_device_name(0))
print(torch.cuda.device_count())

device = "cuda" if torch.cuda.is_available() else "cpu"   # GPU 사용 가능 여부에 따라 device 정보 저장

def conv_relu(in_ch, out_ch, size=3, rate=1):
    conv_relu = nn.Sequential(nn.Conv2d(in_ch, 
                                        out_ch, 
                                        kernel_size=size, 
                                        stride=1,
                                        padding=rate,
                                        dilation=rate),
                               nn.ReLU())
    return conv_relu

class DeepLabLargeFOV(nn.Module):
    def __init__(self):
        super(DeepLabLargeFOV, self).__init__()
        #conv1 ~ conv5
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
        self.features5 = nn.Sequential(conv_relu(512, 512, 3, rate=2),
                                      conv_relu(512, 512, 3, rate=2),
                                      conv_relu(512, 512, 3, rate=2),
                                      nn.MaxPool2d(3, stride=1, padding=1))
                                      #nn.AvgPool2d(3, stride=1, padding=1))
        # FC6
        self.fc6 = nn.Sequential(conv_relu(512, 1024, 3, rate=12), 
                                 nn.Dropout2d(0.5))
        # FC7
        self.fc7 = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=1),
                                 nn.Dropout2d(0.5))

    def forward(self, x):
        out = self.features1(x)
        out = self.features2(out)
        out = self.features3(out)
        out = self.features4(out)
        out = self.features5(out)
        out = self.fc6(out)
        out = self.fc7(out)
        return out
```

### Attention To Scale

---

- Backbone + Attention To Scale

![Untitled](code%20implementation%20ac14e91201d34deea2b5cbc8abe0f82b/Untitled.png)

```python
class AttentionToScale(nn.Module):
    def __init__(self, num_classes, scale, backbone):
        """
        args : 
            num_classes : 21 (type : int)
            scale : [0.5, 1] (type : list)
            backbone : DeepLabLargeFOV (type : class)
        """
        super(AttentionToScale, self).__init__()
        self.scale = scale
        
        # conv1 ~ FC7
        self.backbones = [backbone for i in scale]
        
        # Score Map I = FC8 (f_{i,c}^s)
        self.score_map1 = [nn.Sequential(conv_relu(1024, num_classes, rate=1)) for i in scale]
        
        # Attention to Scale
        # Score Map II
        self.score_map2 = nn.Sequential(conv_relu(1024*len(scale), 512, 3, rate=1),
                                        nn.Dropout2d(0.5),
                                        nn.Conv2d(512, len(scale), kernel_size=1)
                                        )
        # Weights Maps
        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        batch_size, _, feature_map_h, feature_map_w = x.size()
        
        attention_input_list = []
        score_map1_list = []
        
        for i, sc in enumerate(self.scale):
            new_hw = [int(feature_map_h*sc), int(feature_map_w*sc)]
            
            # Scaling assumes bilinear interpolation
            scaled_x = F.interpolate(x, new_hw, mode='bilinear', align_corners=True)
            scaled_x = self.backbones[i](scaled_x)
            
            # Attention input
            scaled_attention_input = F.interpolate(scaled_x, [feature_map_h, feature_map_w], mode='bilinear', align_corners=True)
            attention_input_list.append(scaled_attention_input)
            
            # Score Map I 
            scaled_score_map1 = self.score_map1[i](scaled_x) # E-sup 구현 시, scaled_score_map1를 list로 return 하면 Ok
            rescaled_score_map1 = F.interpolate(scaled_score_map1, [feature_map_h, feature_map_w], mode='bilinear', align_corners=True)
            score_map1_list.append(rescaled_score_map1)
        
        # Concat
        attention_input = torch.cat((attention_input_list), dim=1)
        
        # Get Score Map II
        score_map2 = self.score_map2(attention_input)
        
        # Get Weight Maps
        weights = self.softmax(score_map2)

        # Get Merge two scales
        temp_scales_stack = torch.stack([score_map1*weights[:, idx].resize(batch_size, 1, feature_map_h, feature_map_w) for idx, score_map1 in enumerate(score_map1_list)])
        out = torch.sum(temp_scales_stack, dim=0)
        
        return out
```