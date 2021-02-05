# Multi-Scale Context Aggregation by Dilated Convolutions (DilatedNet) Review

- papers : https://arxiv.org/pdf/1511.07122.pdf

## 0. Abstract 

- Dense prediction 문제는 일반적으로 Image Classficiation과는 다릅니다. 
- Dense prediction 문제에 적합한 새로운 Convolutional Network Module을 제안합니다. 
- 제안된 모듈인 Dilated Convolution은 해상도를 잃지 않고 다양한 크기의 contextual information을 통합합니다. 
- 특히 Receptive field를 지수적으로 증가시키면서도 해상도를 잃지 않습니다. 
- 위의 방법을 통해서 Semantic Segmentation 분야에서 SOTA를 달성할 수 있었습니다. 

## 1. Introduction 

- Semantic Segmentation은 다양한 크기의 상황을 추론해야하고 픽셀단위의 분류를 해야하기에 어렵습니다. 
  - DeconvNet은 위의 문제를 해결하기위해 up-convolutions을 반복해서 다양한 크기의 상황을 추론하고 해상도를 복원했습니다. 
  - 또 다른 방법은 다양한 크기의 입력을 받아서 이를 결합하는 방식입니다. 
- 하지만 이와같은 방법들은 하나의 의문점들을 남기는게 "Down sampling"과 "Rescaled Image"가 필요한지에 대한 의문을 남깁니다. 
- DilatedNet에서는 위의 문제를 해결하고자 Down sampling과 Rescaled Image들을 제거합니다. 
- 그리고 Dilated라는 Convolution을 여러개 결합해서 Down sampling과 Rescaled Input 없이도 효과적인 결과를 가져옵니다. 

## 2. Dilated Convolutions 

![Image for post](https://miro.medium.com/max/658/1*mlHFvK6H_wMCyURSZNZWGQ.png)

- 왼쪽은 일반적은 Convolution을 의미하고 오른쪽은 Dilated Convolution을 의미합니다. 
  - Convolution과 Dilated Convolution의 가장 큰 차이는 Kernel의 t에 l이라는 값이 붙어있는 점입니다. 
  - 이 값에 의해서 둘다 동일하게 3x3의 필터를 가지지만 Receptive Field는 3x3과 5x5로 차이가 발생합니다. 

![Image for post](https://miro.medium.com/max/988/1*btockft7dtKyzwXqfq70_w.gif)



## 3. Multi-Scale Context Aggregation 

![image-20210204145337769](C:\Users\지뇽쿤\AppData\Roaming\Typora\typora-user-images\image-20210204145337769.png)





