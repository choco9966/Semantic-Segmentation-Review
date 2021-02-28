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

![Image for post](https://drive.google.com/uc?export=view&id=1f3IfdgpOVJWS6nUXpdMVYd3uzACqxuF7)

- 왼쪽은 일반적은 Convolution을 의미하고 오른쪽은 Dilated Convolution을 의미합니다. 
  - Convolution과 Dilated Convolution의 가장 큰 차이는 Kernel의 t에 l이라는 값이 붙어있는 점입니다. 
  - 이 값에 의해서 둘다 동일하게 3x3의 필터를 가지지만 Receptive Field는 3x3과 5x5로 차이가 발생합니다. 

![Image for post](https://drive.google.com/uc?export=view&id=1geLZH_nPYp_OJ86gcI8ha--powr3D3A1)

- 위의 수식이 가지는 의미를 한번 예시로 살펴보도록 하겠습니다. 아래의 내용에 있는 수식과 그림은 [안성호님의 블로그의 내용의 사진](http://www.songho.ca/dsp/convolution/convolution2d_example.html)을 참고하였습니다. 

  ![Definition of 2D convolution](https://drive.google.com/uc?export=view&id=1ZFdEsJz2mGTwCpNnHyOZ2GyEcRWULMBA)

- 여기서 y는 output, x는 input, h는 kernel을 의미합니다. 실제 아래의 예시에 대해서 위의 수식이 어떤 값을 가지는지 확인해보겠습니다. 

![image-20210206174332859](https://drive.google.com/uc?export=view&id=1BmLeac42wHc0qbdesRYGunFxg2TZoCVp)

- output -13은 아래와 같은 과정을 통해서 나오게 됩니다. 

![image-20210206174345252](https://drive.google.com/uc?export=view&id=1-J0lNXXKzPxZAp5tNVoeVoq7ihdvSHml)

- 한번 같은 과정을 Dilated가 2인 경우에 대해서 적용해보겠습니다. 

![image-20210206174407172](https://drive.google.com/uc?export=view&id=1SUXQXCbReO3WoH994T1UPUw6st5Xom_R)

- 위와 같은 Dilated 과정을 통해서 Receptive Field가 넓어지는 과정은 아래와 같습니다. 

![image-20210206175311423](https://drive.google.com/uc?export=view&id=1j91Ij_0cpgY_hmvLmRjSECtro_BLB7Ao)

- (a) : F<sub>0</sub> (receptive filed, **green**) → "3x3 filter with 1-dilated convolution" (parameter 9개, **red points**) → F<sub>1</sub>
- (b) : F<sub>1</sub> (receptive filed, **green**)→ "3x3 filter with 2-dilated convolution, padding 1" (parameter 9개, **red points**) ≒ 7x7 filter → F<sub>2</sub>
- (c) : F<sub>2</sub> (receptive filed, **green**)→ "3x3 filter with 4-dilated convolution, padding 3" (parameter 9개, **red points**) ≒ 15 x 15 filter → F<sub>3</sub>
- Convolution의 경우 파라미터가 갯수가 선형으로 증가하지만, Dilated Convolution은 지수적으로 증가하므로 효율적입니다.
  - F<sub>i+1</sub> = 2<sup>i+2</sup> -1 * 2<sup>i+2</sup> -1 의 Receptive Field를 가집니다. 



## 3. Multi-Scale Context Aggregation 

- Context module : multi-scale contextual information을 집계하여 dense prediction 구조의 성능을 높이기 위함입니다. 
- input of context module : font-end(e.g. vgg16)를 통해 해상도가 64x64의 feature map 입니다. 
- Layer 1 ~ Layer 7 : 3x3 convolution with diffrent dilation 을 사용합니다. 
- Layer 8 : 1x1 convolution with 1-dilation 을 사용합니다. 
- truncation : colvolution 이후에 activation function을 ReLU 사용합니다. 

![image-20210204145337769](https://drive.google.com/uc?export=view&id=1C4kK5I__amtwTpil4IigX3r1ar4y1kLp)

- Receptive Field의 경우 원본 이미지를 중심으로 계산하기에 Layer1과 2의 경우 Dilation이 1으로 동일해도 크기가 다릅니다. 아래의 그림에서처럼 Layer 2의 경우에 대해서는 이미 원본 이미지를 피쳐맵으로 바꾼 데이터에 대해 적용됩니다. 

![image-20210206184620570](https://drive.google.com/uc?export=view&id=1Gj676fyGBo4Bd890OvFVyVTfO1dMnct0)

- 논문에서는 Initialization (Le, Quoc V., Jaitly, Navdeep, and Hinton, Geoffrey E. A simple way to initialize recurrent networks of rectified linear units. arXiv:1504.00941, 2015) 방식을 사용했습니다. 

![image-20210206175709833](https://drive.google.com/uc?export=view&id=1pWsFGXrsBRd5uehjRilqaOztFwYrW_OY)

- **a** : the index of the input feature map
- **b** : the index of the output map
- 정확한 수식이 의미하는 바는 모르겠지만 아래의 코드를 함께봤을때 이전의 Weight값을 그대로 가져와서 초기화시키지 않은가 생각은 듭니다. (확실하지는 않습니다)

```python
                L.Convolution(
                    prev_layer,
                    param=[dict(lr_mult=1, decay_mult=1),
                           dict(lr_mult=2, decay_mult=0)],
                    convolution_param=dict(
                        num_output=num_classes * multiplier, kernel_size=3,
                        dilation=dilation, pad=dilation,
                        weight_filler=dict(type='identity',
                                           num_groups=num_classes,
                                           std=0.01 / multiplier),
                        bias_filler=dict(type='constant', value=0))))
```



## 4. Front END 

- a front-end prediction module : VGG-16 네트워크를 사용합니다. 
- 마지막 두개의 layer에 존재하는 pooing 및 striding 제거합니다. 
- 마지막 layer를 제외한 모든 layer의 convolution 연산은 2-dilated 적용합니다. 
- 마지막 layer의 convolution 은 4-dialted 적용합니다. 
- convolution을 바꾸면서 기존 학습된 weight를 모두 초기화시켜야 했지만, 고해상도의 output을 얻을 수 있습니다.
- https://blog.kakaocdn.net/dn/XGBYY/btqV19gdFgI/wxWK9K9pERO4qeaMvTxt11/img.png

<figure> 
    <img src='https://drive.google.com/uc?export=view&id=1p9bfesQP_1pHtCtWjNm0qWho903kZOVH' />
    <figcaption><div style="text-align:center">Padding of 1 adds an extra layers on top of the input matrix. Left: Zero padding. Middle: Reflection padding. Right: Replication padding.
</div></figcaption>
</figure>



### Training 

- SGD
- minibatch size : 14
- learning rate : 10<sup>-3</sup>
- momentum : 0.9
- iteration : 60K

<figure> 
    <img src='https://drive.google.com/uc?export=view&id=1iaXiihE_GBsAFNURsNkGfLoWgBajXu_f' /><br>
    <figcaption><div style="text-align:center">Semantic segmentations produced by different adaptations of the VGG-16 classification network.
</div></figcaption>
</figure>

<figure> 
    <img src='https://drive.google.com/uc?export=view&id=1SiU_zFWNU2DbPqHFsqrYJVMmFSLTzo17' /><br>
    <figcaption><div style="text-align:center">Our front-end prediction module is simpler and more accurate than prior models. This table reports accuracy on the VOC-2012 test set.</div></figcaption>
</figure>



## 5. Experiments 

<figure> 
    <img src='https://drive.google.com/uc?export=view&id=1aIQt7PTuJwy2f6lI1m6gnsFctU9VSSbL' /><br>
    <figcaption><div style="text-align:center">Semantic segmentations produced by different models.</div></figcaption>
</figure>

<figure> 
    <img src='https://drive.google.com/uc?export=view&id=19LgLEfY7To167AtbzhnTWyQT84A19ZFu' /><br>
    <figcaption><div style="text-align:center">Controlled evaluation of the effect of the context module on the accuracy of three different architectures for semantic segmentation.</div></figcaption>
</figure>

<figure> 
    <img src='https://drive.google.com/uc?export=view&id=1F6QoDinG0nb_xuTRQ8NU2PuuybJ3Vu5c' /><br>
    <figcaption><div style="text-align:center">Evaluation on the VOC-2012 test set. ‘DeepLab++’ stands for DeepLab-CRF-COCO-LargeFOV and ‘DeepLab-MSc++’ stands for DeepLab-MSc-CRF-LargeFOV-COCO-CrossJoint (Chen et al., 2015a).</div></figcaption>
</figure>



## 6. Conclusion 



참고자료 

- https://daljoong2.tistory.com/181
- http://towardsdatascience.com/review-dilated-convolution-semantic-segmentation-9d5a5bd768f5
- https://m.blog.naver.com/PostView.nhn?blogId=sogangori&logNo=220952339643&proxyReferer=https:%2F%2Fwww.google.com%2F

