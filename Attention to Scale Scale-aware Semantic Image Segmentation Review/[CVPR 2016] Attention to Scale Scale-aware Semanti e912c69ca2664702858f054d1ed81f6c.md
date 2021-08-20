# [CVPR 2016] Attention to Scale: Scale-aware Semantic Image Segmentation

[https://arxiv.org/pdf/1511.03339.pdf](https://arxiv.org/pdf/1511.03339.pdf)

## Abstract

---

- **Incorporating multi-scale features** in **FCNs** has been a **key element** to achieving state-of-the-art performance on semantic image segmentation.

    ![%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled.png](%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled.png)

- Another way : **Extract multi-scale features** is to feed multiple resized input images to a shared deep network and then merge the resulting features for pixel-wise classification.
- We propose **an attention mechanism** that learns to softly weight the multi-scale features at each pixel location. We adapt a state-of-the-art semantic image segmentation model, which we jointly train with multi-scale input images and the attention model.
- The proposed **attention model** not only **outperforms average and max-pooling**, but allows us to diagnostically visualize **the importance of features at different positions and scales.**
- Moreover, we show that adding **extra supervision** to the output at each scale is essential to achieving excellent performance when merging multi-scale features. We demonstrate the effectiveness of our model with extensive experiments on three challenging datasets
    - PASCAL-Person-Part
    - PASCAL VOC 2012
    - a subset of MS-COCO 2014.

- **Introduction**
    - Various methods based on FCNs → bench marks (2016)
    - contribution → using the use of multi-scale features
    - **two types** of network structures that exploit **multi-scale features**

        ![%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%201.png](%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%201.png)

        1. **skip-net**
            - **define :** combines features from the intermediate layers of FCNs
            - Features within a skip-net are multi-scale in nature due to the increasingly large receptive field sizes.
            - During training, a skip-net usually employs a **two-step process** where it first **trains the deep network backbone** and then **fixes or slightly fine-tunes during multi-scale feature extraction.**
            - **Problem** : two-step training process is not ideal / training time ↑ (e.g. 3~5 days)
        2. **share-net**
            - **define** : resizes the input image to several scales and passes each through a shared deep network.
            - It then computes the final prediction based on the fusion of the resulting multi-scale features
            - A share-net does not need the two-step training process mentioned above (one-step training)
            - It usually employs average-pooling or max-pooling over scales
            - Features at each scale are either equally important or sparsely selected.

        - **Attention models**
            - Recently, **attention models** have shown great success in several CV and NLP
                - Reference : [https://www.youtube.com/watch?v=WsQLdu2JMgI](https://www.youtube.com/watch?v=WsQLdu2JMgI)
                - **seq2seq**

                    ![%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%202.png](%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%202.png)

                - **problem**

                    ![%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%203.png](%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%203.png)

                    ![%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%204.png](%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%204.png)

                - how to solve the problem.

                    → **Attention models** 

                    > D. Bahdanau, K. Cho, and Y. Bengio. Neural machine translation by jointly learning to align and translate. In ICLR, 2015

                    ![%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%205.png](%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%205.png)

                    ![%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%206.png](%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%206.png)

                    ![%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%207.png](%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%207.png)

                    ![%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%208.png](%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%208.png)

                    ![%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%209.png](%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%209.png)

        - Rather than **compressing an entire image or sequence into a static representation**, **attention** allows **the model to focus on the most relevant features as needed**
        - we incorporate an attention model for semantic image segmentation
        - Unlike previous work that employs attention models in the 2D spatial and/or temporal dimension, **we explore its effect in the scale dimension**

            > attention models in the 2D spatial and/or temporal dimension 배경 부족

        - The proposed attention model learns to **weight the multi-scale features according to the object scales presented in the image**

            (e.g. the model learns to put large weights on features at coarse scale for large objects)

            ![%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2010.png](%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2010.png)

        - For each scale, the attention model **outputs a weight map** which weights features pixel by pixel, and the weighted sum of FCN-produced score maps across all scales is then used for classification

        - introduce extra supervision to the output of FCNs at each scale, which we
        find essential for a better performance.
        - We jointly train **the attention model** and the **multi-scale networks**
        - The attention component also gives a **non-trivial improvement** over average-pooling and max-pooling methods.
        - More importantly, the proposed **attention model provides diagnostic visualization**, **unveiling the black box network** **operation** by **visualizing the importance of features at each scale** for every image position.
- **Related Work**
    - **Deep networks : FCNs, DeepLab, ...**
    - **Multi-scale features**
        - **skip-net type : FCN-8s, DeepLab-MSc, ParseNet**
        - **share-net type : CRF, ...**
    - **Attention models for deep networks :**
        - classification : ...
        - detection : ...
        - image captioning and video captioning :
        - NLP : attention
    - **Attention to scale :** To merge the predictions from multi-scale features, there are two common approachs
        - **average-pooling : ...**
        - **max-pooling : ...**
        - We propose to jointly learn an attention model that softly weights the features from different input scales when predicting the semantic label of a pixel.

- **Model**
    - **Review of DeepLab v1**

        ![%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2011.png](%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2011.png)

        - a variant of FCNs (beased on VGG-16) &16-layers
        - FC6 layer에는 dilated convolution (rate = 12)을 (즉, atrous algorithm) 적용하므로 receptive field가 커져서 Field-Of-View is larger라고 불림

    - **Attention model for scales**
        - we discuss how to merge the multi-scale features for our proposed model
        - **model**

            ![%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2012.png](%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2012.png)

        - **input & output of attention model**

            ![%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2013.png](%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2013.png)

        - **score map**

            ![%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2014.png](%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2014.png)

        > Q&A : $\omega_i^2$ 의 크기와 $\omega_i^1$  크기가 다른데, $f_{i,c}^2 (size = f_{i,c}^1)$ 와 어떻게 계산해야할지?

        - 방법 1 :  $\omega_i^1$ 를  $\omega_i^2$와 같도록 Bi-linear interpolation
        - 방법 2 : score map에서 Bi-linear interpolation 을 통해 $f_{i,c}^2 = f_{i,c}^1 (size)$ 같게 하는 부분이 틀렸다?

        - $w_i^s$ 에 대한 분석 & 의미
            - the importance of feature at position $i$ and scale $s$
            - how much attention to pay to features at different positions and scales by visualization
            - Case study (scale을 적용하는 방식?)
                - average-pooling
                - max-pooling
                - attention (this paper)

                ![%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2015.png](%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2015.png)

        - We emphasize that the attention model computes a soft weight for each scale and position, and it allows the gradient of the loss function to be back-propagated through

    - **Extra supervision**
        - loss : Cross-entropy
        - optimization : SGD
        - backbone : Network parameter are initialized from the **ImageNet-pretrained VGG-16 model**

        - **Supervision**을 더 추가하여 스케일별로 출력된 최종 결과에 cross entropy loss를 적용하여 총 1 + S개의 cross entropy를 사용
        - GT는 출력 크기에 맞게 다운샘플링하여 사용

            ![%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2016.png](%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2016.png)

- **Experimental Evaluations**
    - Training : SGD with mini-batch
        - batch-size = 30 images
        - learning rate = 0.001 (multiplied by 0.1 after 2000 iterations)
        - momentum = 0.9
        - weight decay = 0.0005
        - Fine-tuning → 21 hours on NVIDIA Tesla K40 GPU
        - the total training time is twice that of a vanilla DeepLab-LargeFOV
            - all scaled inputs and performs training jointly ($S = 2$)
    - Evaluation metric : IoU
    - Reproducibility : Caffe framework → torch code X (ㅠㅠ)
    - Experiments for contribution
        1. multi-scale inputs : $s \in \{1, 0.75, 0.5\}$
        2. different methods : different methods to merge multi-scale features
            1. average-pooling
            2. max-pooling
            3. attention model

        3. training with or without extra supervision : 

    - **PASCAL-Person-Part**
        - we **focus on the person part** for the dataset, which **contains more training data and large scale variation**
        - Specifically, the dataset contains detailed part annotations for every person, including eyes, nose, etc.
        - training / validation : 1716 images / 1817 images
        - Improvement over DeepLab (validation set)

            ![%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2017.png](%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2017.png)

        - max-pooling  경우 → scales 를 3으로 늘렸더니 성능 증가 (Robust)
        - averge-pooling 및 attention 의 경우 → scales 를 3으로 늘렸더니 성능 감소...
        - However, No matter how many scales are used, our attention model yields better results than average-pooling and max-pooling (Attention is good!)

        ![%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2018.png](%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2018.png)

        ![%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2019.png](%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2019.png)

        - **Failure modes :** The failure examples are due to the extremely difficult human poses or the confusion between cloth and person parts.
            - The first problem may be resolved by acquiring more data, while the second one is challenging because person parts are usually covered by clothes.

    - **PASCAL VOC 2012**
        - The PASCAL VOC 2012 segmentation benchmark consists of **20 foreground object classes and one background class**

            ![%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2020.png](%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2020.png)

        - Pretrained with ImageNet
        - Improvement over DeepLab (**validation set**)

            ![%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2021.png](%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2021.png)

            - PASCAL-Person-Part 과 비슷한 결과 패턴
                - Max-pooing은 scales에 대해 robust하게 성능 증가하지만, average-pooling 및 attention은 오히려 성능 감소
                - 그럼에도 불구하고, attention 사용하면, max-pooling/average-pooling보다 항상 성능 좋게 나옴
                - 추가적으로 , Extra supervision 사용시 성능은 더 증가
        - **the test set for our best model**

            ![%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2022.png](%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2022.png)

            - Attention+ = Attention + E-supv
            - Attention-DT = Attention + "a discriminatively trained domain transform"
            - 한계점 : attention + CRF + pretrained 기법을 사용해도 DPN, Adelaide 를 넘기지 못함

    - **Subset of MS-COCO**
        - **80 foreground object classes and one background class**
        - training / validation : 80K / 40K → random select → 10K / 1.5K \
        - Improvement over DeepLab (validation)

            ![%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2023.png](%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2023.png)

        - class가 많기 때문에 scale을 더 다양하게 할수록 max-pooling/average-pooling은 성능 증가, 반대로 attention은 scale을 더 다양하게 할수록 성능 감소
        - 앞선 다른 데이터셋과 동일하게, Scales + E-supv + attention을 결합하면 성능 증가 (시너지 효과로 봐야하나..)
        - Person class IoU 결

        ![%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2024.png](%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2024.png)

        ![%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2025.png](%5BCVPR%202016%5D%20Attention%20to%20Scale%20Scale-aware%20Semanti%20e912c69ca2664702858f054d1ed81f6c/Untitled%2025.png)

- **Conclusion**
    - For semantic segmentation, this paper adapts a state-ofthe-art model (i.e., DeepLab LargeFOV) to exploit multi-scale inputs.
    - (1) Using **multi-scale inputs** yields **better performance** than a single scale input.
    - (2) Merging the **multi-scale features** with the **proposed attention model** not only **improves the performance** over average- or max-pooling baselines, but also allows us to **diagnostically visualize** **the importance of features** at different positions and scales.
    - (3) Excellent performance can be obtained by adding **extra supervision** to the final output of networks for each scale

- **Attention to Scale: Scale-aware Semantic Image Segmentation (PyTorch version) code**

    [code implementation ](https://www.notion.so/code-implementation-c9c0dd2a82234488be018c61643ebbfd)

## Reference

---

- [http://liangchiehchen.com/projects/DeepLab.html#domain transform](http://liangchiehchen.com/projects/DeepLab.html#domain%20transform)
- [https://openaccess.thecvf.com/content_cvpr_2016/papers/Chen_Attention_to_Scale_CVPR_2016_paper.pdf](https://openaccess.thecvf.com/content_cvpr_2016/papers/Chen_Attention_to_Scale_CVPR_2016_paper.pdf)
- [http://www.navisphere.net/7543/attention-to-scale-scale-aware-semantic-image-segmentation/](http://www.navisphere.net/7543/attention-to-scale-scale-aware-semantic-image-segmentation/)
- [https://ezyang.github.io/convolution-visualizer/index.html](https://ezyang.github.io/convolution-visualizer/index.html)
- [https://yangyi02.github.io/research/attention_scale/attention_scale_slides.pdf](https://yangyi02.github.io/research/attention_scale/attention_scale_slides.pdf)
- [https://distill.pub/2016/deconv-checkerboard/](https://distill.pub/2016/deconv-checkerboard/)