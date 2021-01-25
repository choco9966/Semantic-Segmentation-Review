# SegNet : A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation

-   paper : [https://arxiv.org/pdf/1511.00561.pdf](https://arxiv.org/pdf/1511.00561.pdf)

## 0\. Abstract

1.  VGG16 에서 FC-Layer 3개를 제외한 13개의 Layer으로 구성된 Encoder를 사용
2.  Encoder와 정반대의 Decoder를 이용해서 resolution을 증가시킴
    -   이때, UnMaxpool을 이용해서 non-linear한 Upsampling을 진행함
    -   Unpooling의 경우 학습할 필요가 없어서 파라미터 및 속도의 장점이 있음
    -   UnMaxpool 만 사용할 경우에 Upsampled maps이 Sparse한 단점이 있어서 Conv를 같이 사용해서 Dense하게 만듬
3.  FCN / DeepLab-LargeFOV, DeconvNet과 비교
    -   메모리 및 정확도, 속도의 관점
4.  특히, scene understanding의 분야에서 동기를 얻고 이에 대한 테스크를 수행하려고 함
    -   Scene understanding application 분야는 속도와 메모리가 중요
5.  다른 모델에 비해 파라미터가 적어서 속도, 메모리 측면에서 효율적임

## 1\. Introduction

두 가지의 동기를 통해서 SegNet이라는 모델의 아키텍처를 고안했음

1.  Max Pooling 및 Sub Sampling을 통해서 줄어든 특징 맵의 해상도를 증가시킬지에 대한 고민
2.  Road Scene Understanding applications라는 분야에서 Semantic Segmentation을 수행하기 위해서 모델이 필요한 능력에 대한 고민

특히, Road Scene Undestanding 분야의 경우 몇가지 특징이 존재함

1.  Scene understanding에서 objects간의 관계를 이해 하는데에 Semantic segmentation 사용
2.  Scene understanding의 경우 appearance(road, building)과 shape(cars, pedestrians)를 이해하고 그들간의 관계를 이해할 필요가 있음 (road – cars, pedestrians – side-walk)  
    ![image-20210125005100661](https://drive.google.com/uc?export=view&id=16o4YLm843HOunMFVPdMyzLQL_Tpg_ESh)

## 3\. Architecture



![image-20210125005146494](https://drive.google.com/uc?export=view&id=1sYQHFDQijeNFo9CCcHmAe4EDq1uzIphK)

1.  VGG16의 13개 층을 Encoder로 사용하고 뒤집은 부분을 Decoder로 사용
    -   중간의 Fully Connected Layer 부분을 제거해서 학습 파라미터를 줄임 (134M -> 14.7M)
2.  Enocoder 부분은 pretrained 된 네트워크를 사용
    -   Conv + BN + ReLU
3.  MaxPooling의 장점과 단점
    -   Max Pooling – 2x2 window with stride 2 (overlapped 안되도록)
    -   Max Pooling을 통해 translation invariance + large context를 each pixel에 담음
    -   spatial resolution이 사라지는 문제가 발생 (디테일한 부분이 사라짐)
    -   위의 문제점을 잡아 주기위한 장치가 필요하고 이게 UnMaxPooling 기법
4.  UnMaxPooling의 장점과 단점
    -   메모리의 효율성 + 학습 및 인퍼런스 속도가 빠름
    -   경계에 대한 정보를 효율적으로 저장할 수 있음
5.  다른 Architectures과의 비교 (DeconvNet / UNet)
    -   DeconvNet : FC Layer + 2 stage training + larger parameterization
    -   UNet : Use Feature map / don’t use pretrained weight of VGG net

### 3.1 Decoder Variants

![image-20210125005447762](https://drive.google.com/uc?export=view&id=1U3KiGGJxXo7lzZkXIVzJsaTKdxPZyPOO)

1.  SegNet의 small version으로 4 encoders와 4 decoders으로 구성 (SegNet은 5개씩으로 구성)
2.  Encoders는 max-pooling 과 sub-sampling을 수행하고 이때의 indice를 받아서 unmaxsampling을 수행
3.  Conv layer 이후에는 BN 이 사용되고 Bias는 없음
4.  Decoder Network에는 Relu, Bias 를 사용하지 않음
5.  7x7 의 conv을 이용해서 wide context를 잡으려고함

![image-20210125005812669](https://drive.google.com/uc?export=view&id=1plUchw1aMn8i8foV8EiSETCsiSFwJEBB)

1.  SegNet-Basic과 Encoder는 동일하게 사용
2.  Decoder에는 UnMaxPooling이 아닌 Transposed Convolution으로 전부 대체

### 3.2 Training

![image-20210125010114862](https://drive.google.com/uc?export=view&id=1KBO2iQhkWlqVAFQuAIJsolhznBO7iCxW)

1.  Camvid Dataset
    -   Training Dataset : 367
    -   Test Dataset : 233
    -   Class : 12 (배경포함 )
2.  Initialized (He initialized)
3.  Optimization : SGD (lr : 0.1, momentum 0.9)
4.  validation score가 가장 높은 Epoch 선택
5.  Cross Entropy Loss
6.  Medium Frequency Balancing
    -   larger Class의 weight를 1보다 작게 설정

### 3.3 Analysis

0.  실험을 위한 평가함수와 테스트 데이터셋 세팅
    -   Performance Metric
        -   Global Accuracy (Pixel Accuracy - G)
        -   Class average accuracy (Mean Pixel Accuracy - C)
        -   mean Intersection over union (mIoU)
        -   Boundary F1-measure (BF)
            -   mIoU의 경우 계산방법과 사람이 인식하는 경계부분이 좀 다른데, 이를 보완하기 위해 경계부분에 대해서 정확도를 측정하는 방법
    -   Test Set (CAMVID)
        -   CamVid validation set을 기준으로 global accuracy가 가장 높은 Epoch로 추론
        -   하지만, 도로와 건물, 하늘, 보도처럼 대부분의 이미지를 차지하는 클래스 때매 클래스의 평균 정확도가 높다고해서 Global Accuracy가 높은게 아님

![image-20210125011316214](https://drive.google.com/uc?export=view&id=1wrOoau6ppofAe589YNBdC5nv6Efl5vt_)

1.  3가지 경우에 대해서 각각 성능을 비교 (파라미터 / 추론속도 / 성능)
    -   unsampling의 종류 (Bilinear / Upsampling – indices / Learning to upsample)
    -   Median frequency balancing
    -   Natural frequency balancing
2.  Bilinear-Interpolation without any learning performs의 경우



![image-20210125011429289](https://drive.google.com/uc?export=view&id=1YfJ7ZqxF8HWWZSMfsx382YGJoE-_dgva)

3.  SegNet-Basic vs FCN-Basic

![image-20210125011450220](https://drive.google.com/uc?export=view&id=16kOGG_XbHRMarL250SahqFFeCTGNohu8)

-   SegNet-Basic : Less memory (Storage multiplier)
-   FCN-Basic : encoder feature maps
    -   encoder 층마다 feature maps 필요
    -   Faster Inference time (Deconvolution layer가 적음)

4.  Others



![image-20210125011842347](https://drive.google.com/uc?export=view&id=1zqNVWXRclKZhrDxHgiPjLMQ1fW7pykfF)

-   SegNet-Basic-SingleChannelDecoder : 추론속도와 파라미터의 수가 엄청 작음
-   SegNet-Basic-EncoderAddition : Unsampling시에 Max-pooling Indices를 사용
-   FCN-Basic-NoDimReduction : 성능이 가장 높고 Infer time도 작음. 파라미터와 메모리의 경우 1.625M과 64로 가장 큼
-   FCN-Basic-NoAddition-NoDimReduction : Addition을 제거시에 정확도가 많이 감소 84.8 -> 67.8
-   SegNet-Basic-EncoderAddition, FCN-Basic-NoDimReduction과 같이 무거운 모델이 성능이 높음
    -   FCN에서는 차원축소를 제거하는게 성능과 BF 측면에서 성능향상이 큼
    -   Memory와 정확도 사이에는 Trade-off 관계를 보임

5.  Summary
    -   인코더 기능 맵이 완전히 저장되어 있을 때 최상의 성능을 얻을 수 있음. 특히 의미론적 등고선 설명 메트릭(BF)에 확하게 반영됨
    -   추론 중 메모리가 제한될 때, 압축된 형태의 인코더 특징 맵(차원 감소, 최대 풀링 지수)을 적절한 디코더(예: SegNet 유형)와 함께 저장 및 사용하여 성능을 개선할 수 있음
    -   Decoder의 깊이가 커지면 성능이 향상함

## 4\. Benchmarking

### 4.1 Road Scene Segmentation

![image-20210125012527739](https://drive.google.com/uc?export=view&id=10Hvyfhh1nEhMC44uX4Mfnv3jCaq4Tykb)

![image-20210125012546836](https://drive.google.com/uc?export=view&id=1GMBfFVur9dKBI3lv67Un3licuasxHEvb)

-   FCN과 DeconvNet에 비해서 SegNet과 DeepLabv1은 적은 iteration에서 높은 성능을 보임
    -   40K, 80K 에서 성능의 차이가 특히 발생하고 >80K에서는 DeconvNet하고 성능은 비슷한 모습을 보이고, 오히려 BF 부분은 DeconvNet의 성능이 더 높음
    -   SegNet과 DeepLabv1은 G, C, mIoU에 대해서는 초기에는 성능이 거의 비슷하고 Iteration이 늘어날 수록 차이가 발생 (BF의 경우 40K에서도 SegNet이 우수)

![image-20210125012532699](https://drive.google.com/uc?export=view&id=17PFTCCeNfIGX9sicy0p4xlYDf_Wxk-3z)

### 4.2 SUN RGB-D Indoor Scenes

![image-20210125012649178](https://drive.google.com/uc?export=view&id=1vR1WP3DWmTOsFP0HrlSmxjcJUij6P7oC)

-   Deep Architectures (SegNet, DeconvNet)가 80K에서 낮은 성능을 보임 (G, C, mIoU)
-   SegNet의 경우 G와 BF가 DeepLab-LargeFOV 보다 높은 경향을 보이지만 C와 mIoU는 높은 경향을 보이고 CamVid 데이터셋에 비해 성능이 많이 감소함
    -   첫번째 원인은 Class의 수가 증가했고, small class가 이미지에 많이 등장하는 경향이 있음
    -   두번째 원인은 VGG를 사용하는 Deep한 Architecture때문에 발생 (파라미터가 많아서 수렴 x, 정보손실이 큼)
    
    

![image-20210125012717342](https://drive.google.com/uc?export=view&id=1EPpz3_0ITxssksasDhJy3nov5euQjx4u)

## 5\. Discussion and Future Work

1.  보통의 Deep Learning은 모델이 깊고, 데이터가 많고, 학습이 길어지면 성능이 좋아짐
    -   실험 결과처럼 학습 시간의 상승이 성능의 향상이 크게 관련 없을경우 학습 시간이 중요함(?)
    -   인퍼런스 시간과 메모리 역시 AR, 자율주행과 같은 특수한 분야에서 매우 중요
    -   전반적인 효율성(성능, 메모리, 학습 및 테스트 시간)에서 SegNet은 효율적임
2.  Pascal, MS-COCO 처럼 Class가 적은 경우보다 Scene segmentation은 Class가 훨씬 많고 동시에 등장해서 더 어려움
3.  속도 및 메모리, 성능에 대해서 동일한 비교를 하기 위해서 End-to-End의 학습 및 추론을 사용 (DeconvNet의 경우 Instance-wise Segmentation을 하는데 그렇지 않았다는 의미같음)

## 6\. Conclusion

1.  road and indoor scene understanding 분야에서 동기를 얻어서 memory와 computational time을 효율적으로 만드려고 시도
    
2.  SegNet을 다른 논문인 FCN, DeconvNet 등과 비교해서 architectures에 따라서 어떤 식으로 정확도, 학습 및 추론 속도 등이 trade-offs 관계를 가지는지 파악
    
3.  Encoder network feature maps를 저장해서 활용하는게 성능은 가장 좋지만 inference time과 memory 측면에서 좋지 않음
    
4.  위의 3의 성능을 보완하기위해서 max-pooling indices를 사용하면 충분히 좋은 성능에 도달
    
5.  추후, End-to-End의 학습이 잘 되도록 개선할 예정
    
6.  SegNet은 FCN, DeepLab v1 보다는 느리지만 DeconvNet 보다는 빠름
    
    
    
    ![image-20210125012924245](https://drive.google.com/uc?export=view&id=1D-8aYzvjxIQjUa8KcnjlNHWdatiw2D8A)
    
    -   FCN, DeepLabv1에 비해 SegNet은 Decoder Architecture가 있어서 느릴 수밖에 없음
    -   DeconvNet에서 FC Layer를 제거한 구조여서 DeconvNet 보다는 속도가 빠름
    
7. SegNet은 Traning, Inference memory가 작은 편이고 Model Size 또한 FCN, DeconvNet에 비해 작음

8. Object의 크기가 큰 경우에 대해서는 잘 맞추는 모습을 보이지만, 반대의 경우에 대해서는 성능이 떨어지는 모습을 보임

   

   ![image-20210125012944893](https://drive.google.com/uc?export=view&id=192p_jZzTsu_q__Y_nUtEWxSxwuM4_0dw)


### 6.1 Advantages

1.  Scene Understanding이라는 특수한 분야의 Task에 초점을 맞춰서 문제를 해결하려고함
    -   Object의 크기가 크다는 점
    -   Object간에 동시에 발생하고 서로 관계가 있다는 점 (보행자-인도, 자동차-도로)
    -   자율주행의 경우 Inference 속도가 빨라야 한다는 점
2.  구조 자체가 DeconvNet에서 Fully Connected Layer만 뺀 구조라서 SegNet의 장점을 어필하기 위해 다양한 시도와 초점을 맞춰서 진행
    -   Scene 분야 / Memory 및 Training, Inference Time / Parameter에 의한 오버피팅
    -   DeconvNet vs SegNet / FCN vs SegNet / Unet vs Segnet
    -   Iterations이 적은 경우와 많은 경우에 대해 성능이 어떻게 나오는지 비교
    -   클래스의 크기가 작은 경우와 많은 경우에 대해 성능이 어떻게 나오는지 비교

### 6.2 Disadvantages

1.  DeconvNet에 Fully Connected Layer만 뺀 논문치고는 굉장히 화려한게 아닌가 생각이 듬 (시점으로보면 미리 준비하고 있었는데 DeconvNet이 나와서 arxiv에 낸게 아닌가 불쌍하기도함)
    
2.  CamVid와 SUNRGB-D 라는 특수한 데이터셋에 대해서는 성능이 제일 좋았지만 다른 데이터셋에 대해서는 어떻게 나왔을지 비교가 필요(일반화성능)
    
    -   참고로 FCN, DeconvNet, DeepLab 논문의 경우 위의 2개 데이터에 대해서 실험을 한 내용이 없어서 직접 구현해서 돌렸을텐데 튜닝을 어느정도 했을지에 대한 의문도 남음
    -   PASCAL VOC 2012에 대해서도 비교를 해야 정확한 결과였을 것 같음
    -   Global Accuracy가 가장 높은거로 선택했다고 했는데 mIoU나 다른 지표로 weight를 선택해서 실험했을때 어떤 결과가 나왔을지에 대해 궁금
3.  Large Object를 잡고 Object간의 관계를 파악하는 것을 수행하는게 MaxPooling -> UnMaxPooling인데 과연 이걸로 충분한가? 사실 DeconvNet도 위의 장치가 있는게 결과 차이가 너무 나는게 이상함. Fully Connected Layer 있고 없고에 대해서 어떤 차이가 나오는지 등에 대해 정확한 분석이 들어가야 되고 위의 원인이 맞는지도 의심스러움
    
4.  같은 얘기 너무 반복함. 모든 내용에 Training Time / Computation / Memory 중요하다는 얘기가 반복되고 FCN이나 다른 방법에 비해 크게 상승했는지도 의문임
    
5.  CamVid Data가 가지고 있는 문제점
    
    -   SegNet의 실험 대부분이 CamVid에서 비교했는데 해당 데이터는 약간의 문제가 있음
        
        -   Input / Output data가 Time Correlated됨
        
        
        
        ![image-20210125014232260](https://drive.google.com/uc?export=view&id=13cObpRnnos4YyEDJAuSQBhNwIkI6691k)
        
        -   Dataset의 수가 작고 이상한 라벨들이 존재
            -   Train 367 / Test 233
            -   Weird Label이 존재 (Bycycle == Person / 라벨이 자세하지 않음)

![image-20210125013424071](https://drive.google.com/uc?export=view&id=1cDWtjfDZkaFQgwBTXdg5-DzQdklJCv9D)

## 7\. Appendix

### 7.1 참고자료

1.  J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In CVPR, 2015
2.  A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In NIPS, 2012
3.  K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. CoRR, abs/1409.1556, 2014
4.  C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. CoRR, abs/1409.4842, 2014
5.  B. Hariharan, P. Arbelaez, L. Bourdev, S. Maji, and J. Malik. ´ Semantic contours from inverse detectors. In ICCV, 2011
6.  C. L. Zitnick and P. Dollar. Edge boxes: Locating object ´ proposals from edges. In ECCV, 2014
7.  [https://medium.com/@sunnerli/simple-introduction-about-hourglass-like-model-11ee7c30138](https://medium.com/@sunnerli/simple-introduction-about-hourglass-like-model-11ee7c30138)

### 7.2 기존의 네트워크와의 비교



![image-20210125013713262](https://drive.google.com/uc?export=view&id=1Rjm8f0XOYS05KcRIioAp0kTMRzulITi3)

이미지 출처 : [https://medium.com/@sunnerli/simple-introduction-about-hourglass-like-model-11ee7c30138](https://medium.com/@sunnerli/simple-introduction-about-hourglass-like-model-11ee7c30138)