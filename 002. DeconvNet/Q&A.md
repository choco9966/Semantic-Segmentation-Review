# Q&A 

### 1. DeconvNet 같은 논문에서 VGG의 Fully Connected Layer를 1x1 Convolution 혹은 7x7 Convolution으로 변경했지만, 표기는 fc라든지 Fully Connected Layer으로 표기하는데 그 이유가 무엇인가요??  

정확한 이유는 모르겠지만 VGG에서 Fully Connected Layer라고 표현하고 이를 Convolution으로만 바꿨기에 Fully Connected라고 그냥 표기하지 않았나 생각이듭니다. 실제 원 저자의 코드를 봐도 Convolution으로 구현된 것을 볼 수 있습니다. 

```
# 7 x 7
# fc6
layers { bottom: 'pool5' top: 'fc6' name: 'fc6' type: CONVOLUTION
  blobs_lr: 1 blobs_lr: 2 weight_decay: 1 weight_decay: 0
  convolution_param { kernel_size: 7 num_output: 4096 } }
layers { bottom: 'fc6' top: 'fc6' name: 'bnfc6' type: BN
  bn_param { scale_filler { type: 'constant' value: 1 }
             shift_filler { type: 'constant' value: 0.001 }
             bn_mode: INFERENCE } }
layers {  bottom: "fc6"  top: "fc6"  name: "relu6"  type: RELU}

# 1 x 1
# fc7
layers { bottom: 'fc6' top: 'fc7' name: 'fc7' type: CONVOLUTION
  blobs_lr: 1 blobs_lr: 2 weight_decay: 1 weight_decay: 0
  convolution_param { kernel_size: 1 num_output: 4096 } }
layers { bottom: 'fc7' top: 'fc7' name: 'bnfc7' type: BN
  bn_param { scale_filler { type: 'constant' value: 1 }
             shift_filler { type: 'constant' value: 0.001 }
             bn_mode: INFERENCE } }
layers {  bottom: "fc7"  top: "fc7"  name: "relu7"  type: RELU}
```

### 2. DeconvNet에서 1x1 1x1 -> 7x7 이렇게 된 이유가 무엇인가요? 그리고 deconv-fc6의 채널이 512인 이유가 무엇인가요? 

![figure3](https://drive.google.com/uc?export=view&id=1tF4Gpc9WskzkuKZ4a9Zahf57Dx5QioNB)

일단, fc6와 fc7의 kernel size는 VGG를 그대로 가져왔고 1x1은 Output size의 결과입니다. deconv-fc6의 채널이 512인 이유는 코드의 저자가 직접 설정한 값입니다. 

### 3. DeconvNet에서 어떠한 방식으로 Region-Proposals를 추출한 것인가요? 

![](https://drive.google.com/uc?export=view&id=1nxpPorNlv85wSNBbdSS4w9Icdh9ozSKC)

논문에서 자세히는 아니지만 [Edge Box](https://pdollar.github.io/files/papers/ZitnickDollarECCV14edgeBoxes.pdf)이라는 논문의 방식을 사용했다라고 합니다. 해당 논문에서는 슬라이딩 윈도우 방식과는 다르게 Edge를 이용해서 바운딩 박스 후보를 생성하고 해당 영역 내에서만 추가 분류 작업을 진행하는 방식입니다. 이러한 후보의 생성의 방식은 아래와 같습니다. 

1. [Structured Edge detector](https://arxiv.org/pdf/1406.5549.pdf)를 이용해서 에지 픽셀에 대한 에지 반응을 계산 

![image-20210123202258631](C:\Users\지뇽쿤\AppData\Roaming\Typora\typora-user-images\image-20210123202258631.png)



1. 에지 픽셀에 대한 에지 반응을 계산 
2. 

![img](http://amroamroamro.github.io/mexopencv/opencv_contrib/structured_edge_detection_demo_01.png)



참고자료 

- https://donghwa-kim.github.io/EdgeBoxes.html
- http://www.navisphere.net/5473/edge-boxes-locating-object-proposals-from-edges/



### 4.  Although batch normalization helps escape local optima, the space of semantic segmentation is still very large compared to the number of training examples and the benefit to use a deconvolution network for instance-wise segmentation would be cancelled. Then, we employ a two-stage training method to address this issue, where we train the network with easy examples first and fine-tune the trained network with more challenging examples later.

일반적인 Classificiation에서의 Batch Normalization의 효과를 생각해보면 하나의 이미지가 하나의 클래스를 대변하는 경우입니다. 하지만 Semantic Segmentation의 경우 하나의 이미지에 하나의 Object가 있지 않습니다. 여러개의 Object가 있고 각각에 대해서 다른 정보를 포함하고 있습니다. 그렇기에 하나의 이미지를 통째로 Batch Normalization을 하게되는 방식은 Semantic Segmentation에서는 별로 좋지 못한 방법입니다. 이를 해결하기 위해 2 Stage의 방식을 도입해서 1 Stage에서는 학습된 사진에 하나의 Object가 있는 쉬운 예제들로 학습을 진행하고 2 Stage에서 어려운 예제로 Fine Tuning을 하는게 합리적이라는 의미입니다. 

### 5. Unpooling과 Pooling의 경우 대칭의 형태를 가지는 것인가요? 

![figure3](https://drive.google.com/uc?export=view&id=1tF4Gpc9WskzkuKZ4a9Zahf57Dx5QioNB)

그림의 이미지 크기와 코드를 같이 보면 이해가 쉬울 것 같습니다. 첫번째로 등장하는 Maxpooling의 경우 마지막 Unpooling과 대응되게 됩니다. 이렇게 되어야만 하는 이유는 이미지의 크기가 같아야하고 해당 피처 맵에서 정보를 뽑았기 때문입니다. 실제 코드로도 보면 1번째 Max Pooling에서 나온 인덱스가 마지막 Un MaxPooling의 인덱스로 활용되고 있습니다. 

```python
# https://github.com/choco9966/Semantic-Segmentation-Review/blob/main/002.%20DeconvNet/code/DeconvNet%20(VOC%20Format).ipynb
    def forward(self, x):
        h = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        h = self.relu1_2(self.bn1_2(self.conv1_2(h)))
        h, pool1_indices = self.pool1(h)
        
        h = self.relu2_1(self.bn2_1(self.conv2_1(h)))
        h = self.relu2_2(self.bn2_2(self.conv2_2(h)))
        h, pool2_indices = self.pool2(h)
        
        h = self.relu3_1(self.bn3_1(self.conv3_1(h)))
        h = self.relu3_2(self.bn3_2(self.conv3_2(h)))
        h = self.relu3_3(self.bn3_3(self.conv3_3(h)))
        h, pool3_indices = self.pool3(h)
        
        h = self.relu4_1(self.bn4_1(self.conv4_1(h)))
        h = self.relu4_2(self.bn4_2(self.conv4_2(h)))
        h = self.relu4_3(self.bn4_3(self.conv4_3(h)))
        h, pool4_indices = self.pool4(h)
        
        h = self.relu5_1(self.bn5_1(self.conv5_1(h)))
        h = self.relu5_2(self.bn5_2(self.conv5_2(h)))
        h = self.relu5_3(self.bn5_3(self.conv5_3(h)))
        h, pool5_indices = self.pool5(h)
        
        h = self.relu6(self.bn6(self.fc6(h)))
        h = self.drop6(h)

        h = self.relu7(self.bn7(self.fc7(h)))
        h = self.drop7(h)
        
        h = self.debn6(self.deconv6(h))
        
        h = self.unpool5(h, pool5_indices)
        h = self.debn5_3(self.deconv5_3(h))
        h = self.debn5_2(self.deconv5_2(h))
        h = self.debn5_1(self.deconv5_1(h))
        
        h = self.unpool4(h, pool4_indices)
        h = self.debn4_3(self.deconv4_3(h))
        h = self.debn4_2(self.deconv4_2(h))
        h = self.debn4_1(self.deconv4_1(h))
        
        h = self.unpool3(h, pool3_indices)
        h = self.debn3_3(self.deconv3_3(h))
        h = self.debn3_2(self.deconv3_2(h))
        h = self.debn3_1(self.deconv3_1(h))
        
        h = self.unpool2(h, pool2_indices)
        h = self.debn2_2(self.deconv2_2(h))
        h = self.debn2_1(self.deconv2_1(h))
        
        h = self.unpool1(h, pool1_indices)
        h = self.debn1_2(self.deconv1_2(h))
        h = self.debn1_1(self.deconv1_1(h))
        h = self.score_fr(h)        
        return torch.sigmoid(h)
```

### 6. DeconvNet에서 Transposed Convolution의 경우 이미지의 크기를 변경하지 않는데 사용하는이유? 사실 Convolution Layer 또한 해당 문제를 해결할 수 있었을텐데 굳이 Transposed Convolution을 사용한 이유가 있는지? 



### 7. Fixed Size receptive Field에 의해서 FCN이 단점이 있었고 이를 DeconvNet에서 해결했다고 하는데 어떤 방식으로 된지 이해가 잘 안갑니다. 

![](https://drive.google.com/uc?export=view&id=1JM0QWgdrUDfYvI9JLNhLCg-EIZVTTJv8)

먼저 Receptive Field에 대해서 고민을 해보면 객체의 Feature Map을 해석하기 위해서 얼마만큼의 영역을 통해서 해석하는지로 생각할 수 있습니다. VGG 논문에서 이에 대한 내용이 잘 나옵니다. 

![](https://drive.google.com/uc?export=view&id=1y3IlcOB8axVD-fbGjKe44FMyXkVq7Jzx)

해당 논문에서는 위의 그림에서와 같이 7x7 Filter를 통해서 7x7의 영역을 통해서 해석하는 것을 3x3 4개와 MaxPooling을 통해서 같은 효과를 낼 수 있다고 표현합니다. 

![](https://drive.google.com/uc?export=view&id=1w5k2PxSvIpUgJKa8fn7nAjkE5bV2MHFQ)

이처럼 FCN, DeconvNet, SegNet 모두 3x3의 Filter를 통해서 Feature를 추출하는데 그렇게 할 경우에 아래와 같은 문제점이 있습니다. 아래의 사진에서 (a)의 그림처럼 버스의 경우 유리에 자전거가 비친 모습을 가지는데, 이러한 부분적인 특징들만 가지고 학습을 진행하게 되면 추론시에 해당 영역에 bicycle이 등장할 수 있게 되는 것입니다. 후속 논문들에 대해서는 위와 같은 문제를 해결하기 위해서 다양한 방식이 도입되었습니다. 

- 배경과 객체를 함께 특징을 추출하는 방법 : [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)
- 효과적으로 Receptive Field의 크기를 키우는 방법 : [Large Kernel Matters - CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2017/papers/Peng_Large_Kernel_Matters_CVPR_2017_paper.pdf), [DilatedNet](https://arxiv.org/abs/1511.07122)
- 입력 이미지의 특성에 따라서 필터의 모양을 유기적으로 변형시키는 방법 : [Deformable Convolutional Networks](https://arxiv.org/abs/1703.06211)

참고자료 

- https://medium.com/@msmapark2/vgg16-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-very-deep-convolutional-networks-for-large-scale-image-recognition-6f748235242a

  