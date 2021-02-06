# Q&A

### 1. SegNet에서 Single Channel Decoder가 의미하는게 무엇인가요? 

### 2. The final decoder output is fed to a multi-class soft-max classifier to produce class probabilities for each pixel independently. 이라는 표현이 나오는데 multi-class soft max 를 사용하는 것의 의미가 무엇인지? 

답변 : 각각의 클래스 채널에 대해서 softmax를 취한 것을 의미합니다. 이에 대한 내용은 다음의 [링크](https://github.com/choco9966/Semantic-Segmentation-Review/blob/main/A%20Deep%20Convolutional%20Encoder-Decoder%20Architecture%20for%20Image%20Segmentation%20(SegNet)%20Review/%EC%8B%9C%EA%B7%B8%EB%AA%A8%EC%9D%B4%EB%93%9C%20vs%20%EC%86%8C%ED%94%84%ED%8A%B8%EB%A7%A5%EC%8A%A4_%EC%9D%B4%EB%AA%85%EC%98%A4.pdf)의 자료를 참고하면 좋습니다. 

- 출처 : https://discuss.pytorch.org/t/binary-segmentation-bcewithlogitsloss-pos-weight/71667

