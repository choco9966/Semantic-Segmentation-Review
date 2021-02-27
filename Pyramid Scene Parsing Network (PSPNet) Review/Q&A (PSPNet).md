## Q&A (PSPNet)

Q1. PSPNet에서 ImageNet 실험에 대해서 앙상블을 하는 부분이 있는데 어떤 모델들을 앙상블 하였는지:? 

A1. 해당 부분을 http://image-net.org/challenges/talks/2016/SenseCUSceneParsing.pdf의 pdf에서 확인한 결과 두가지 중 하나로 판단됩니다. 

첫째, 5개의 ResNet 모델을 앙상블한 것으로 판단되지만 정확하지는 않습니다. 

```markdown
| Pretrained Model  | Result      |
|-------------------|-------------|
| Resnet50          | 40.11/79.55 |
| Resnet101         | 41.29/80.04 |
| Resnet152         | 42.23/80.46 |
| Resnet256 + SS    | 43.39/80.90 |
| Resnet256 + MS    | 44.59/81.80 |
| Ensemble 5 Models | 45.85/82.48 |
```

- But it is time consuming and only useful for competitions

둘째, 이제까지 실험한 5가지의 경우를 앙상블한 경우 

<figure>
    <img src='https://drive.google.com/uc?export=view&id=1ebIrvmjwGhXLWrktpSFX6IYkxVASPfl1'/>
      <figcaption><div style="text-align:center">출처 : http://image-net.org/challenges/talks/2016/SenseCUSceneParsing.pdf
  </div></figcaption>
</figure>



Q2. PSPNet에서 ImageNet 실험에 대해서 앙상블을 하는 부분이 있는데 어떤 모델들을 앙상블 하였는지:? 

> In a deep neural network, the size of receptive field can roughly indicates how much we use context information. Although theoretically the receptive field of ResNet [13] is already larger than the input image, it is shown by Zhou et al. [42] that the empirical receptive field of CNN is much smaller than the theoretical one especially on high-level layers



<figure>
    <img src='https://drive.google.com/uc?export=view&id=174t2gwj2XfIH-QXF18Pr_8B0Gf_UfxOt' /><br>
      <figcaption><div style="text-align:center">출처 : Zhou et al. OBJECT DETECTORS EMERGE IN DEEP SCENE CNNS
  </div></figcaption>
</figure>

A. Zhou et al. [42]의 논문을 보면 이론적인 Receptive Field의 경우 Pooling이 들어갈때마다 넓어지는 모습을 보이고 있습니다. (Pool5 : Theoretic size) 하지만, 실제 activation maps가 어디서 활성화 되었는지를 살펴보면 국소적인 영역에 한정된 것을 볼 수 있습니다. PSPNet에서는 이러한 문제를 제기하면서 Global Average Pooling을 통해서 강제적으로 전체를 보게하는 방법이 필요하다는 것을 강조하는 것으로 보입니다. 

Q3. PPM에서 Pooling의 결과를 1x1 Convolution을 통해서 채널을 줄이는 이유가 무엇인가요? 

<figure>
    <img src='https://drive.google.com/uc?export=view&id=1c5H4QcP8lYhX69vJo2h4EMH_Q_1OJ51p' /><br>
      <figcaption><div style="text-align:center">출처 : ICNet for Real-Time Semantic Segmentation on High-Resolution Images
  </div></figcaption>
</figure>

A. ICNet에서 분석한 결과에 따르면 해당 부분에 대해서 Channel이 급격히 증가하는 경우 Inference Time이 급격하게 증가합니다. 1/N으로 줄여도 Inference Time이 이렇게 증가하는데 만일 그대로 사용했으면 기존대비해서 많이 증가했을거라고 생각합니다. 추가로 PPM의 결과와 Feature Map의 채널결과가 동일한데 두가지 특징에 대해 균형을 맞추려는 의도도 있었을 것 같습니다. 즉, Inference Time을 줄이기 위해서와 기존의 Feature Map과 균형을 맞추기위해서 이러한 작업을하지 않았을까 생각합니다. 