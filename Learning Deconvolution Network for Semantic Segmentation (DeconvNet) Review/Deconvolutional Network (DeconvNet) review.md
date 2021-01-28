# Deconvolutional Network (DeconvNet) review 

## Abstract 

1. 기존의 Fully Convolutional Network가 가지고 있는 한계점을 극복하기위해 Layer를 더 깊게 쌓음 
2. unpooling layers에서 Maxpooling 과 Transposed Convoltuion을 같이 도입 
3. 위의 결과, Detail한 측면과 Multi Scales한 측면에서 기존 대비 많은 효과가 있었음  

## Fully Convolutional Networks의 한계  

- 기존의 FCN은 아래의 한계점을 보유하고 있음 
  1. 네트워크는 기존에 정의된 고정된 Receptive field를 가짐. 그렇기에 object의 크기가 receptive field 대비 크거나 작은 경우에 대해서 잘 못맞추는 경향을 보임. 즉, object가 큰 경우에 대해서는 object의 부분적인 정보들만을 가지고 분류하게 되는 문제점이 발생 (예: 버스를 분류할때, 버스 전체를 보는 것이 아니라 앞면, 유리 등 부분적인 특징을 가지고 분류하게 됨)

     ![figure1](https://drive.google.com/uc?export=view&id=142BAw0S7mrjRHtSu6maBUObTRmandkUc)

  2. 작은 object에 대해서는 일관성없게 분류하는 문제점을 가지고 있음. 종종 무시하거나 배경으로 예측

     - 위의 한계점은 사실 Skip Architecture으로 극복하려고 했던 점들이지만, 충분한 해결책이 아니고 details 과 semantics 측면에서 trade - off를 가짐 

       ![figure2](https://drive.google.com/uc?export=view&id=1CMNufuZbtGK7VBIZ_jgW6rqKhaTQGq86)

  3. object의 detail한 구조를 잃어버리거나 포괄적인 모습으로 대체함

     - 이러한 이유는 Deconvolutional Layer (Transposed Convolution)의 특성상 coarse하고 simple하기에 발생 

- 이러한 한계점을 극복하기 위해서 Deconvolutional Netwokrs 에서는 아래의 방법들을 시도 

  1. Deep한 Deconvolution Networks를 생성. 이는 Deconvolution, unpooling, ReLU 등의 layers으로 구성되어 있음 
  2. trained 된 네트워크를 개개인의 object proposals에 적용함으로서 scale 문제에서 자유롭게 함 

  그 결과, PASCAL VOC 2012에서 높은 정확도를 얻었고, FCN 기반의 모델과 앙상블시에 더 좋은 성과를 얻었음 

## Deconvolutional Network의 구조 

### Architecture 

![figure3](https://drive.google.com/uc?export=view&id=1tF4Gpc9WskzkuKZ4a9Zahf57Dx5QioNB)

- Convolution : input image의 feature를 추출해서 multidimensional feature representation으로 변환함 
  - 위의 구성은 VGG 16-layer net에서 마지막 classification layer를 제거한 것이고, 결국 Convolution networks는 13개의 convolution layers으로 구성되어있고 ReLU 와 Pooling이 convolutions 사이에 수행됨. 마지막으로 2개의 fully connnect layers (1x1 convolution)으로 class-specific projection을 수행함 
- Deconvolution : convolution network으로부터 추출된 feature를 object segmentation으로 생성 
  - Convolution Network의 Mirror 버전으로 unpooling, deconvoltuion 그리고 ReLU layers으로 구성되어있음

### Unpooling 

- Convolution에서 Pooling은 대표값을 추출해서 noisy activation을 걸러주는 역할을 하지만, spatial information을 잃어버리는 문제점을 가지고 있음. 이러한 문제를 해결하기 위해서, 아래의 그림과 같은 Pooling 시의 활성화된 위치를 기억하고 해당 값을 복원하는 방법을 사용 
- 이러한 전략은 input object의 구조를 기억하는데 매우 유용함 

![figure4](https://drive.google.com/uc?export=view&id=1X6HueEiVh1WS7RcwsmZXgLkDy2AnfRRC)

- PyTorch에서는 `nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)` 명령어를 통해서 위치를 추출하고, `unpool1 = nn.MaxUnpool2d(2, stride=2), unpool1(h, pool1_indices)` unpooling시에 해당 인덱스를 넣어줌으로서 구현가능함 

### Deconvolution

- unpooling layer의 output은 크지만, sparse activation map 이라는 문제점이 있음(대부분의 값이 0으로 비활성화 되어있음). deconvolution layer는 이러한 sparse를 dense activation map 으로 조밀하게 만드는 특징을 가지고 있음
- 이러한 deconvolutional layers는 unpool layer의 size를 유지하므로, input object의 모양을 재건축한다. 그러므로, convolution network와 유사하게 다양한 level의 모양을 잡는 역할을 수행. lower layers의 필터는 전반적인 모양을 higher layers는 디테일한 모양을 잡는 역할

### Analysis of Deconvolution Network 

![figure5](https://drive.google.com/uc?export=view&id=14HlP3nnpBcjWFmLIfSkoXKixIL970y6D)


- Deconvolution Network의 Deconvolution과 Unpooling에 의해서 활성화된 activation map을 보면 위와 같음  

- (b), (d), (f), (h), (j)와 같은 Deconvolution의 결과는 dense하고 (c), (e), (g), (i)와 같이 Max unpooling의 결과는 sparse함. 또한, lower layer은 전반적인 특징 (location, shape, region)을 잡는 반면 higher layer는 복잡한 패턴을 잡음 

  ![figure6](https://drive.google.com/uc?export=view&id=19Facs8tU9eTLHQxSFlU8A8VkUN4StCof)

- 결국, 위의 그림과 같이 FCN에 비해 디테일한 모습이 많이 살아나는 장점을 보임 

## Training 

### Batch Normalization 

- DNN은 Internal-covariate-shift 때문에 최적화하기 어렵고, 이를 해결하기 위해서 batch normalization을 적용 
- 모든 layer는 표준 정규분포을 통해서 정규화됨 

### Two-stage Training 

- 비록 normalization에 의해 local optimal를 탈출하는데 도움을 주지만, semantic segmentation의 공간은 학습 데이터의 수에 비해서는 크고 instance segmentation을 수행하는 deconvolution의 장점은 사라짐 
- 이를 방지하기 위해서, 2 stage의 학습을 진행
  - 1 stage에서는 쉬운 예제로 train하고 2 stage는 보다 어려운 데이터로 학습을 진행 
  - 1 stage의 쉬운 예제는 object proposals 알고리즘으로 객체가 있을많나 영역을 자르고, 실제 정답 object를 crop하여 이를 중앙으로 하는 bounding box를 만듬 
  - 2 stage는 1 stage에서 잘라낸 이미지들 중 실제 정답을 crop하기 전에 실제 정답과 잘 겹치는 것들을 활용하여 2차 학습을 진행 (실제 정답만 사용하는게 아니라 Iou가 0.5이상인 후보군을 전부 사용)
- 일반적인 이미지 분류는 테스크가 쉽지만, Segmentation은 픽셀단위의 분류를 하기에 Optimal을 찾는 과정이 더 어렵고 local optimal을 탈출하기가 힘듦 
  - 그렇기에, 1stage에서는 하나의 객체만 포함하도록 해서 쉬운 테스크를 진행하고 2stage에서는 좀 더 어려운 이미지로 학습을 진행 
- https://www.facebook.com/groups/TensorFlowKR/permalink/1358930397781348/

## Inference 

- DeconvNet은 개별의 instance에 대해서 semantic segmentation을 수행
- 개별 instance를 생성하기 위해 input image를 window sliding을 통해서 충분한 수의 candidate proposals를 만듬
- 이후, 이에 대해 semantic segmentation을 수행하고 proposals에 대해서 나온 모든 결과를 aggregate해서 전체 이미지에 대한 결과를 생성
- 추가적으로, FCN과 앙상블시에 성능이 향상됨

### Aggregating Instance-wise Segmentation Map

- 몇몇 Proposals는 부정확한 예측을 가지고 있기에, aggregation동안에 suppress 해줌
- Pixel-wise Maximum or average를 통해서 충분히 robust한 결과를 만듬 
- 이후, output map에 fully-connected CRF를 적용 

### Ensemble with FCN

- DeconvNet은 fine-details를 잘 잡는 반면에, FCN은 overall shape를 추출하는데 강점을 가지고 있음 
- instance wise prediction은 object의 various scales을 다루고, FCN은 coarse scale에서의 context를 잡는데 강점이 있음 
- 둘을 독립적으로 시행후에 Ensemble 하고, CRF를 적용하면 가장 좋은 결과가 나옴 

## Experiments 

- PASCAL VOC 2012 segmentation dataset과 Microsoft COCO 으로 실험한 결과 아래와 같이 좋은 결과를 얻음 

- Optimization은 SGD를 이용하고, learning rate, momentum, weight decay는 각각 0.01, 0.9, 0.0005으로 설정. ILSVRC 데이터셋으로 pre-trained된 모델을 사용했으며 deconvolution networks는 zero-mean Gaussians으로 initialize 함. 또한, drop out을 제거하고 batch normalization을 활용  

  ![figure7](https://drive.google.com/uc?export=view&id=1Xnu7z-QKDexF2pmS2I0bixK3RKjEgVwB)

  ![figure9](https://drive.google.com/uc?export=view&id=1TkSAHELTQ4NogLSwj8gOWsgbe8yqBGGZ)

- 또한, proposals의 수를 늘릴 수록 결과가 좋아지는 것을 볼 수 있음 ![figure8](https://drive.google.com/uc?export=view&id=14H551JMJfooLPiXQDOCpQgG0UWTf51GE)

## Discussion 

- 논문의 아키텍처 자체는 Layer를 깊게 쌓고, Transposed Conovlution 뿐만 아니라 기존에 사용하던 방법인 Max Unpooling을 쓴 것으로 굉장히 단순합니다. 그리고 Layer가 굉장히 깊기에 학습속도의 측면에서 굉장히 느려보입니다. (이후, SegNet에서도 언급하지만 1x1 convolution 부분이 굉장히 속도가 느림)
- 하지만, 글을 전개하는 과정이나 효과를 설명하는 부분이 좋았습니다. 
  - 기존의 FCN의 단점이 무엇인지 명확히 명시해주고 해결하는 측면
  - Unpool과 Deconvolution이 어떠한 차이점을 보이는지 같이 사용하면 어떠한 효과가 있는지
- Multi-scale object를 잡기 위해서, window sliding을 통해 객체가 있을법한 위치를 proposal 하지만 Fast-RCNN 계열에서 언급하듯 너무 비싼 코스트를 사용하는 방법 같습니다. 
- CRF, FCN과의 Ensemble 및 다양한 실험이 내용에 포함된 것도 좋았습니다. 
