Q1. Atrous Conv 대신에 Large Conv를 쓰면..? 

Q2. 3x3 Conv -> 1x1 Conv ? 

Q3. MS 의미하는 것 / Flip 의미하는 것 ? 

Q4. 1X1 Global Avg -> Bilinear Interpolation 어떻게 가능한지 

```
import torch
import torch.nn as nn

input = torch.rand([1, 256, 1, 1])
print(input.shape)

m = nn.Upsample(scale_factor=16, mode='bilinear')

print(m(input).shape)
m(input)
```

Q5. Global Avg가 가지는 의미가 위치는 상관없이 객체의 특징만을 추출 ? 