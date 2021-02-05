# Q&A

### 1. Elastic Deformation이 unet에서 처음 나온 방법인가요? 

논문에서 언급되었듯이 Dosovitskiy, A., Springenberg, J.T., Riedmiller, M., Brox, T.: Discriminative unsupervised feature learning with convolutional neural networks. In: NIPS (2014) 에 나온 방법을 사용하였습니다. 

### 2. U-Net에서 적은 이미지를 가지고도 잘 학습할 수 있고 Overlap Tite 및 Deformation 등의 방법을 썻다고 나오는데 네트워크에 자체에 의해서 모델의 성능을 높이는 장치가 있는 것인가요? 

개인적인 생각으로는 논문에서는 네트워크 자체보다는 위의 Augmentation을 이용해서 적은 이미지로도 성능을 높였다는 것 같습니다. 하지만, 제가 하고 있는 반도체의 Segmentation 연구에서도 적은 이미지로도 결과가 잘나왔는데 그 이류를 생각해보면 Concatenate 장치를 통해서 다른 모델들에 비해 성능이 잘 나오지 않았을까 생각합니다. 학습을 충분히 하지 않아도 인코더의 피쳐맵을 활용해서 복원하기에 적은 데이터로도 더 효과적인 결과가 나온 것 같습니다.  

### 3. EM 데이터 셋의 경우 원본 이미지가 512x512이고 입력 이미지의 크기는 572로 되었다는데 패치 등의 과정이 어떤식으로 진행되는 것인가요? 

이미지를 Resize한 후에 Overlap Tite 전략을 수행한지 패치로 자르고 Resize한 후에 Overlap Ttie를 수행한지 정확히는 모르겠지만, 리사이즈를 통해서 바꿔서 학습할 필요가 있습니다. 그리고 572->570으로 줄어드는 과정에서 계속해서 미러링 전략을 통해서 크기를 맞춰줄 필요가 있어보입니다. 572를 388으로 줄인 후에 미러링을 했으면 성능이 많이 나빴을 것 같습니다. 

### 4. We generate smooth deformations using random displacement vectors on a coarse 3 by 3 grid . The displacements are sampled from a Gaussian distribution with 10 pixels standard deviation? 이 의미하는 것이 무엇인가요? 

3x3 grid에서 1번째랑 2번째 성분이 휘어지는 성질 그리고 방향등을 통해서 랜덤하게 뽑아서 Deformation을 수행했다고 이해했습니다. 

- https://hj-harry.github.io/HJ-blog/2019/01/30/Elastic-distortion.html



Q1. We generate smooth deformations using random displacement vectors on a coarse 3 by 3 grid . The displacements are sampled from a Gaussian distribution with 10 pixels standard deviation?

Q2. The final decoder output is fed to a multi-class soft-max classifier to produce class probabilities for each pixel independently

