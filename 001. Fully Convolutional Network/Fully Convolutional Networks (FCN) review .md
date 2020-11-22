# Fully Convolutional Networks (FCN) review 

## Abstract

1. AlexNet을 시작으로 하는 CNN 모델들의 발전을 Image Segmentation이라는 영역에 접못한 방법 
2. Fully Convolutional 과 Skip Architecture라는 두가지 방법론을 도입 

## Fully Convolutional 

- 정의 : Fully Connvected Layer를 1x1 Convolution으로 변경
  - 이미지의 위치정보를 기억 
  - 임의의 입력크기에 대해서도 일관성 있는 결과를 생성 (Fully connected layer는 입력의 크기가 동일해야 하는데, Convolution 는 상관없음) 

![figure1](C:\Users\user\Desktop\공부자료\Semantic Segmentation\001. Fully Convolutional Network\figure1.PNG)

- Fully Convolution Layer가 이미지의 위치정보를 해치지 않는 이유 

  ![figure2](C:\Users\user\Desktop\공부자료\Semantic Segmentation\001. Fully Convolutional Network\figure2.PNG)

  - 위의 그림에서 보이듯이 1x1 Convolution은 사각형 형태의 아웃풋이 나오기에 흰색의 특징을 해당 위치에 보존
  - 하지만, Fully Connected는 하나의 배열로 펼치기에 위치 정보를 손실하게 됨 

- Fully Convolution Layer가 임의의 입력크기에 대해서도 일관성 있는 결과를 생성하는 이유 

  ![figure3](C:\Users\user\Desktop\공부자료\Semantic Segmentation\001. Fully Convolutional Network\figure3.PNG)

  - Convolution은 kernel의 파라미터에 의해 영향을 받고, 이미지 혹은 레이어의 크기에 대해서는 상관
    없음  

## Upsampling (Deconvolution, Transposed Convolution) 

- 정의 : Pooling 단계에 의해서 감소한 이미지의 크기를 원본 이미지의 크기로 복원하는 방법 

![figure4](C:\Users\user\Desktop\공부자료\Semantic Segmentation\001. Fully Convolutional Network\figure4.PNG)

- 2x2 크기의 입력값을 2x2의 kernel을 이용해서 3x3의 output을 만듬 

  - [[0, 1] , [2, 3]]의 input이 kernel의 0, 1, 2, 3에 각각 곱해진 후 더해짐 
  - 왼쪽 아래의 그림처럼 초록색으로 겹치는 부분에 대해서는 더해짐 (노랑 * [w3, w6, w9] + 청록 * [w1, w4, w7])

- 오른쪽 아래의 그림처럼 input 간의 간격을 벌림으로서 더 큰 output으로 upsampling 할 수도 있음 (이후에 ASPP Net에서 Dilated Convolution이라는 항목으로 다룰 예정)

- Tranposed Convolution과 Deconvolution의 차이점

  ![figure5](C:\Users\user\Desktop\공부자료\Semantic Segmentation\001. Fully Convolutional Network\figure5.PNG)

  - 가장 주요한 차이점은 Deconvolution은 Convolution의 결과를 입력값과 동일하게 만드는 작업이고, Transposed Conolvution은 크기는 같지만 연산의 결과는 입력값과 다를 수 있다는 점입니다. 

  - Transposed Convolution의 이름이 어떻게 붙게 되었는 지에 대한 과정을 살펴보면 아래와 같습니다. 

    1. Convolution을 행렬로 바꿔서 계산하게 되면 아래와 같은 모습을 가집니다. 

    ![figure6](C:\Users\user\Desktop\공부자료\Semantic Segmentation\001. Fully Convolutional Network\figure6.PNG)

    2. 위의 Convolution을 Transposed 취해서 ouput에 곱해줍니다. 그렇게 하면, 새로운 출력값을 얻을 수 있고 이는 input 값하고는 다른 결과를 가지게 됩니다. ![figure7](C:\Users\user\Desktop\공부자료\Semantic Segmentation\001. Fully Convolutional Network\figure7.PNG)
    3. 참고로 이는 수식적으로 Convolution의 미분값과 동일해서, 아래와 같이 코드상으로 계산이 가능합니다. ![figure8](C:\Users\user\Desktop\공부자료\Semantic Segmentation\001. Fully Convolutional Network\figure8.PNG)

- Upsampling 시에 발생가능한 문제

  - 문제점 : image Segmentation은 원본 이미지와 출력 이미지의 크기가 같아야하는데, Pooling과 Upsampling 과정에서 이미지의 크기가 달라질 수 있음 (2의 배수가 되지 않는 경우, 2로 나누어 떨어지지 않아서 버림하거나 올림하기에 발생)

    ![figure9](C:\Users\user\Desktop\공부자료\Semantic Segmentation\001. Fully Convolutional Network\figure9.PNG)

  - 해결책 : Zero padding을 통해서 2의 배수를 만들어서 해결 

    ![figure10](C:\Users\user\Desktop\공부자료\Semantic Segmentation\001. Fully Convolutional Network\figure10.PNG)

## Skip Architecture 

- 정의 : ResNet의 아이디어처럼 얕은 층의 특징과 깊은 층의 특징을 결합하려는 시도 
- 얕은 층은 일반적으로 복잡하지 않은 정보를 가지고, 깊은 층은 복잡한 정보를 가지기에 앙상블의 효과가 있음 

![figure11](C:\Users\user\Desktop\공부자료\Semantic Segmentation\001. Fully Convolutional Network\figure11.PNG)

- FCN의 저자는 이러한 방법을어느 깊이까지의 얕은 정보를 활용하는지에 대해서 실험함 
  - FCN-32s : 사용 x 
  - FCN-16s : Pooling Layer 4번째의 결과를 결합 
  - FCN-8s : Pooling Layer 3, 4번째의 결과를 결합 

![figure12](C:\Users\user\Desktop\공부자료\Semantic Segmentation\001. Fully Convolutional Network\figure12.PNG)

## Results 

![figure13](C:\Users\user\Desktop\공부자료\Semantic Segmentation\001. Fully Convolutional Network\figure13.PNG)

- 결과적으로 위의 그림처럼, 다양한 Pooling의 결과를 쓰면 점수가 더 좋아지고 Backbone으로 VGG16을 사용했을 때 결과가 가장 좋았습니다. 

- Image Segmentation에서 딥러닝을 활용한 초기의 논문이었고, Resnet의 기술이라든지 다양한 방향성을 제시해준 논문이어서 굉장히 가치가 있었습니다. 하지만, 위의 그림에서 보이듯이 아직 개선의 여지는 많습니다. 

  1. 디테일한 모습을 살리지 못했다. 
  2. 오른쪽 라이더의 얼굴부분과 자전거 부분에 비는 픽셀이 존재한다. (부분적인 특성만 봐서 예측하지 못함)

- 다음 논문인 Deconvolutional Networks (DeconvNet)에서는 이러한 한계점을 어떻게 극복했는 지에 대해 보도록 하겠습니다. 

  ​