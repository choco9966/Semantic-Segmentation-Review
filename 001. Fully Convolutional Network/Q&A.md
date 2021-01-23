# Q&A 

### 1. 다른 논문에서 FCN에서 Bilinear Interpolation 방법을 사용했다고 하는데 실제로는 Transposed Convolution을 사용합니다. DeconvNet, SegNet 논문에서 이렇게 표현하는 이유는 무엇인가요? 

- FCN 32s, 16s, 8s 모두 Transposed Convolution을 사용하고 Bilinear Interpolation으로 Upsampling 하지는 않습니다. 하지만 아래의 그림처럼 16s와 8s는 2x의 Upsampling 과정이 있고 이 역시 Tranposed Convolution으로 진행됩니다. 하지만, 해당 Convolution의 가중치를 세팅할때 Bilinear interpolation으로 만든 weight initialize를 거치는데 이 때문에 후속 논문에서 Bilinear Interpolation 방법으로 Upsampling 했다고 표현하지 않았나 생각이 듭니다. 