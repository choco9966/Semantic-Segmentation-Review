## DeepLabv1 Q&A

Q1. 에너지함수와 최적화의 object함수와의 차이점은 무엇인가요? 패널티항의 유무인가요?

A1. 

Q2. Class score maps are quite smooth -> bilinear interpolation ot increase their resolution ...? 

A2. 

Q3. 정확도면에서는 Smooth 

A3. 

Q4. CRF를 직관적으로 설명하면 어떻게 설명할 수 있는지? 

A4. 두개가 같은 객체인지 확인해서 같은 

원본 이미지의 색깔하고 위치 부분이 들어가서 해결? -> 

Mean Iou 측면에서는 왜 높게..? 전에 SegNet할때 BF Measure 안쓰는지? 

Q5. DCNN 하고 CRF는 END-TO-END 인가요? 

A5. DCNN / CRF는 서로 분리되어서 

CRF 문제점 -> 비행기

Q6. Section3.2 VGG16 기반의 그거와 Ours의 Receptive Field 줄어드는 부분(?)

A6. 메모리만을 목적으로? 

Q7. 

---

코드리뷰 

Q8. Bilinear Interpolation True / False 

https://gaussian37.github.io/dl-pytorch-snippets/