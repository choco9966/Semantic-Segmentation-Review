# Q&A 

### 1. 다른 논문에서 FCN에서 Bilinear Interpolation 방법을 사용했다고 하는데 실제로는 Transposed Convolution을 사용합니다. DeconvNet, SegNet 논문에서 이렇게 표현하는 이유는 무엇인가요? 

FCN 32s, 16s, 8s 모두 Transposed Convolution을 사용하고 Bilinear Interpolation으로 Upsampling 하지는 않습니다. 하지만 아래의 그림처럼 16s와 8s는 2x의 Upsampling 과정이 있고 이 역시 Tranposed Convolution으로 진행됩니다. 하지만, 해당 Convolution의 가중치를 세팅할때 Bilinear interpolation으로 만든 weight initialize를 거치는데 이 때문에 후속 논문에서 Bilinear Interpolation 방법으로 Upsampling 했다고 표현하지 않았나 생각이 듭니다. 

### 2. Sampling in patchwise training can correct class imbalance [27, 7, 2] and mitigate the spatial correlation of dense patches [28, 15]. In fully convolutional training, class balance can also be achieved by weighting the loss, and loss sampling can be used to address spatial correlation. 에서 Fully Convolutional Training을 통해서 어떻게 patchwise training과 같은 효과를 낸지 궁금합니다. 

![](https://drive.google.com/uc?export=view&id=1Ch9XJ82hpBqBj1TTrIhPsnp9UXYxDsTk)

일단, Patchwise 방식을 보면 위의 그림과 같습니다. 이미지에는 불필요한 배경이 너무 많기에 이를 제외시켜주고 필요한 부분만 Patch로 받아서 학습을 하는 방식입니다. 이렇게 하게 되면 크게 2가지의 장점이 있습니다. 

1. Patch에 가중치를 줌으로서 Class Imbalance를 해결할 수 있다. 
2. 필요한 부분만을 학습하기에 수렴이 더 빠르다. 

하지만 Patch의 경우에는 필요한 부분만 Crop하기에 Spatial Correlation을 잃어버리는 단점이 발생합니다. 논문에서는 Fully Convolutional Training에서도 몇가지 테크닉을 도입하면 위의 장점을 가져올 수 있으면서 Spatial Correlatition을 잃지 않는다고 합니다. 먼저 Class에 Weight를 줌으로서 Minor한 Class가 더 잘 맞도록 해당 로스에 가중치를 줄 수 있습니다. 두번째로 이미지의 전체 부분에 대해서 Loss를 계산하는 것이 아니라 임의로 샘플링을 해서 계산할 수 있습니다. 64x64의 이미지의 경우 총 4096개의 Loss가 발생하는데 이를 다 사용하는게 아니라 임의로 샘플링을 해서 사용하는 방법입니다. 이렇게 하게 되면 Loss에 샘플링을 하는것이기에 배경이 이미지의 대부분을 차지할 경우 제거할 수 있고 Spatial Correlation을 잃지도 않습니다. 

![](https://drive.google.com/uc?export=view&id=1w5k2PxSvIpUgJKa8fn7nAjkE5bV2MHFQ)

하지만 (a)의 자동차처럼 이미지의 대부분을 차지하는 버스의 경우에 대해서는 위의 방식이 조금 위험할 수도 있을 거라는 생각이듭니다. 

참고자료 

- https://stats.stackexchange.com/questions/266075/what-is-the-difference-between-patch-wise-training-and-fully-convolutional-train
- https://stackoverflow.com/questions/42636685/patch-wise-training-and-fully-convolutional-training-in-fcn