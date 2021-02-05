# Pyramid Scene Parsing Network (PSPNet) Review 

- papers : https://arxiv.org/pdf/1612.01105.pdf

## 0. Abstract 

- Pyramid Pooling Modules를 이용해서 서로 다른 영역의 정보를 통합하는 PSPNet을 제안합니다. 
- Secne Parsing에서 좋은 성능을 거두었으며 ImageNet scene parsing challenge 2016, PASCAL VOC 2012, Cityscapes에서 좋은 성능을 차지했습니다. 

## 1. Introduction 

- Scene parsing 같은 경우 몇가지 어려운 점이 있습니다. 
  - 의자와 소파, 말과 소 등과 같이 유사한 모습을 가지지만 다른 라벨들이 존재합니다. 
  -  The new ADE20K dataset [43] is the most challenging one with a large and unrestricted open vocabulary and more scene classes. 

![image-20210201175812861](C:\Users\지뇽쿤\AppData\Roaming\Typora\typora-user-images\image-20210201175812861.png)

- FCN이 비록 좋은 성능을 보였지만 몇가지 한계점을 보입니다. 
  - 보트의 경우 물 위에 떠있는 특징이 있지만 외관이 자동차와 비슷해서 잘못 예측하는 문제가 있습니다. 
  - 고층빌딩처럼 객체가 큰 경우에 대해서 비슷한 클래스인 빌딩으로 착각하게 됩니다. 
  - 배게와 같이 침대와 유사한 모습, 색을 가지는 경우 잘못 분류하는 문제가 있습니다. 
- 위와 같은 문제는 객체의 지역 지역적인 정보만을 가지고 예측해서 발생한 문제입니다. 
- 이를 해결하기 위해서는 객체 주변의 전체 정보를 통해서 예측할 필요가 있습니다. 
  - 예를들어, 첫번째 사진의 경우 강 위에 보트가 떠있고 그 주변에 보트를 넣을 공간이 있다라는 "Context"를 통해 예측하면 문제를 해결할 수 있습니다. 
- PSPNet에서는 FCN처럼 지역적인 정보를 이용해서 예측하는 방법에 더해서 global pyramid pooling을 통해 전역적인 정보까지 결합해서 예측하는 방법입니다. 
- 추가적으로 "deeply supervised loss"라는 최적화 전략을 제시해서 좋은 성능을 달성했습니다. 

## 2. Related Work 

## 3. Pyramid Scene Parsing Network 

### 3.1 Important Observation 

### Mismatched Relationship 

### Confusion Categories 

### Inconspicuous Classes





## 4. Deep Supervision for ResNet-Based FCN 



## 5. Experiments 



## 6. Concluding Remarks 









## 참고자료 

[이론] 

- https://www.youtube.com/watch?list=WL&v=siwbdHhQPXE&feature=youtu.be
- https://gaussian37.github.io/vision-segmentation-pspnet/
- https://towardsdatascience.com/review-pspnet-winner-in-ilsvrc-2016-semantic-segmentation-scene-parsing-e089e5df177d

[코드]

- https://github.com/CSAILVision/semantic-segmentation-pytorch
- https://github.com/hszhao/PSPNet