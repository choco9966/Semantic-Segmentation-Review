# Pyramid Scene Parsing Network (PSPNet) Review 

- papers : https://arxiv.org/pdf/1612.01105.pdf

## 0. Abstract 

- Sence parsing의 경우 Open Vocabulary 와 diverse secens 두 가지 어려운 점을 가지고 있습니다. 

- 위의 문제를 해결하기위해서 Pyramid Pooling Modules를 이용한 PSPNet을 제안합니다. 

  - Global context information을 탐색하는 능력을 가집니다. 
  - 서로 다른 영역을 기반으로 하는 Context를 탐색할 수 있습니다. 

- Secne Parsing에서 좋은 성능을 거두었으며 ImageNet scene parsing challenge 2016, PASCAL VOC 2012, Cityscapes에서 좋은 성능을 차지했습니다. 

- 참고로 Open Vocabulary Problem이란 정확히는 모르겠지만 카테고리가 굉장히 많은 경우와 카테고리 사이의 상하위 관계가 있는 문제를 의미하는 것 같습니다. 

  <figure> 
      <img src='https://drive.google.com/uc?export=view&id=1s9LORL59xZU8XfEQhw9Ft0BqC80cl0P2' /><br>
      <figcaption><div style="text-align:center">출처 : Open Vocabulary Scene Parsing
  </div></figcaption>
  </figure>

## 1. Introduction 

- Scene parsing 같은 경우 몇가지 어려운 점이 있습니다. 

  ![figure2](https://drive.google.com/uc?export=view&id=1exvbgb36wk2XqzlZ-wHkEPLYAM2zVDMr)

  - 의자와 소파, 말과 소 등과 같이 유사한 모습을 가지지만 다른 라벨들이 존재합니다. 

    ![image-20210206193424786](https://drive.google.com/uc?export=view&id=1_DxURz5KJcOGPQ12Vagkr_KFewQeZpPC)

  - Global Scene category clues를 통해서 예측하는 능력이 부족

    - 예) 강, 선착장 등을 통해서 보트를 예측 

  - The new ADE20K dataset [43] is the most challenging one with a large and unrestricted open vocabulary and more scene classes. 

![image-20210206193525845](https://drive.google.com/uc?export=view&id=1rqQaG1yaAgyA8i7O8NXcY5bHapS_w7E3)

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

![image-20210206193620991](https://drive.google.com/uc?export=view&id=1_TE-bX0CfTXrvr_pBuYrjOzI3Hz0U9RG)

- Context Relationship을 잘못 파악하는 경우 
  - 복잡한 배경을 이해하기 위해서는 객체들 간의 관계를 파악하는게 중요합니다. 예) 하늘에 비행기가 떠다닌다.
  - 하지만, 위의 경우 보트가 강 위에 떠있음에도 외관이 자동차와 비슷하다는 이유로 잘못 분류한 것을 볼 수 있습니다. 

### Confusion Categories 

![image-20210206193735629](https://drive.google.com/uc?export=view&id=1n77vLmm46pHytFM3Q64OC1sPT9ZdSnv_)

- Confusion Categories를 가지는 경우 
  - 산과 언덕, 들과 땅, 벽과 집, 건물 과 고층 건물처럼 비슷하지만 조금 다른 라벨들이 많고 이를 잘못 분류할 가능성이 높습니다. 
  - FCN의 경우 고층빌딩을 빌딩과 고층빌딩으로 섞어서 분류한 것을 볼 수 있습니다. 
  - 위의 문제들은 범주 간의 관계(고층빌딩 – 하늘)를 통해서 해결할 수 있습니다. 

### Inconspicuous Classes

![image-20210206193810496](https://drive.google.com/uc?export=view&id=1N7296W0jqxa0j9FhrclKrPnhgjlq1CAv)

- Inconspicuous Classes (눈에 띄지 않는 클래스) 
  - 신호등이나 표지판같이 객체들은 중요하지만 작아서 잘 분류되지 않는 문제가 있습니다. 
  - 고층 빌딩과 같이 크기가 큰 객체들은 Receptive Field를 벗어나서 예측이 잘 안되는 문제가 있습니다. 

### Summary 

- 객체마다의 크기가 달라서 서로 다른 Receptive Field가 필요합니다. 
- 그리고 주변 정보(문맥)를 파악해서 객체를 예측하는데 사용해야 합니다.

### 3.2 Pyramid Pooling Module 

![image-20210206193956060](https://drive.google.com/uc?export=view&id=1E9uACmME_ncleJXTtltNdBJOmLRL3dXq)

- 객체마다의 크기가 달라서 서로 다른 Receptive Field가 필요합니다. → 서로 다른 크기의 Receptive Field를 가지는 Pooling
- 그리고 주변 정보(문맥)를 파악해서 객체를 예측하는데 사용해야 합니다. → Global Average Pooling을 도입 

### Global Average Pooling 

- Global Average Pooling (1)

![image-20210206201356924](https://drive.google.com/uc?export=view&id=1kpavzHy5J3n5MMoSt8A_xM9DCdgnStgZ)

- Global Average Pooling (2)

![image-20210206201440726](https://drive.google.com/uc?export=view&id=11SKk_8bBpNtgNWSPQrPnYQG5OJPVNHsd)

- Global Average Pooling vs Convolution 

<figure> 
    <img src='https://drive.google.com/uc?export=view&id=1aiEYE-Ogb38zVDDLFXeIjMNpXV-2_2mS' /><br>
    <figcaption><div style="text-align:center">출처 : https://www.youtube.com/watch?list=WL&v=siwbdHhQPXE&feature=youtu.be (임정근님의 A-GIST 비전&로보틱스 발표)
</div></figcaption>
</figure>

![image-20210206201519467](https://drive.google.com/uc?export=view&id=1ImsyRWyOrhac_TjESDDSLMq6iis8K6D8)

하나의 Global Average Pooling만 적용한 경우 복잡한 ADE20K 데이터 셋에서는 적합하지 않았습니다.

- 위의 문제를 극복하기 위해서 여러 형태의 Receptive Field를 가진 Global Average Pooling을 적용합니다. 
- 그럴 경우 서로 다른 영역의 정보들을 뽑을 수 있고 이를 통합해서 다양한 크기의 Context를 생성할 수 있습니다. 

![image-20210206201533724](https://drive.google.com/uc?export=view&id=1CFTNGjHp_B1MFrT7LY25BF8qxGnwPiqJ)

- PPM의 구조를 자세히 보면 4개의 크기를 가지는 풀링(1, 2, 3, 6)에 의해서 부분 영역들을 추출합니다. 
- 이후, Conv을 통해서 채널의 수를 정제하고 Upsampling으로 크기를 키운 후에 이를 원래의 피처 맵과 결합 함으로서 다양한 크기의 문맥 특징을 추출할 수 있습니다.

![image-20210206201602741](https://drive.google.com/uc?export=view&id=1tGH8B_GgzRxjCbSzd_g_77ypilQhN7_V)

## 4. Deep Supervision for ResNet-Based FCN 

![image-20210206201619826](https://drive.google.com/uc?export=view&id=13I7BoONK1NMQq96KJ_tXJocHM9XSTi2W)

## 5. Experiments 

### 5.1 Implementation Details 

- Augmentation
  - Random mirror 
  - Random Resize 
  - Random Rotation 
  - Gaussian Blur 
  - Crop

### 5.2 ImageNet Scene Parsing Challenge 2016 

### Ablation Study for PSPNet 

![image-20210206201727713](https://drive.google.com/uc?export=view&id=1vQYmlpfFWyEv8fm3PW4wvth_NXSJiE6l)

### Ablation Study for Auxiliary Loss

![image-20210206201734537](https://drive.google.com/uc?export=view&id=1UkAAihNFklIn8ebmGmILoftIWAAlrJLS)

### Ablation Study for Pre-trained Model 

![image-20210206201739301](https://drive.google.com/uc?export=view&id=1WIzOKznmudRBamsu2eJHrKhveoAFCMjH)

### Multi Scale Testing 

<figure> 
    <img src='https://drive.google.com/uc?export=view&id=1klFgnJpg_lj6fJCYvKsv1QSe5_gkTp42' /><br>
    <figcaption><div style="text-align:center">출처 : “Single-Shot Refinement Neural Network for Object Detection”, RefineDet
</div></figcaption>
</figure>

### PASCAL VOC 2012 

![image-20210206201919742](https://drive.google.com/uc?export=view&id=1TMljPrruzWwi6eC9dKA1ikFrvBVfy9oj)

### Cityscapes

![image-20210206201928062](https://drive.google.com/uc?export=view&id=13YYNyycMzBY4uSMCEhlRNvAXVgULt73m)

## 6. Concluding Remarks 

### 6.1 Advantages 

- Global Average Pooling이라는 이미 나온 개념을 가져와서 해당 문제의 특징을 잘 해결한 논문입니다. 
- GAP의 크기를 다양하게 적용해서 다양한 크기의 정보를 추출했습니다. 

### 6.2 Disadvantages

- 속도가 오래 걸리는 문제가 있습니다. (https://stackoverflow.com/questions/61889355/global-average-pooling-does-not-affect-training-speed)



## 참고자료 

[이론] 

- https://www.youtube.com/watch?list=WL&v=siwbdHhQPXE&feature=youtu.be
- https://gaussian37.github.io/vision-segmentation-pspnet/
- https://towardsdatascience.com/review-pspnet-winner-in-ilsvrc-2016-semantic-segmentation-scene-parsing-e089e5df177d

[코드]

- https://github.com/CSAILVision/semantic-segmentation-pytorch
- https://github.com/hszhao/PSPNet