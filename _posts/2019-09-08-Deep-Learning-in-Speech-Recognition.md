---
title: "Deep Learning in Speech Recognition"
excerpt: Stanford Seminar - Deep Learning in Speech Recognition을 들으면서 정리해봤습니다.
date: 2019-09-08
layout: post
category:

- [Speech Recognition]

tag:

- Speech Recognition

---

Speech Recognition에 대한 사전 지식이 없어서, [Stanford Seminar - Deep Learning in Speech Recognition](https://www.youtube.com/watch?v=RBgfLvAOrss) 을 들으면서 정리해봤습니다. 앞 부분은 배경 설명이라 29:52 부터 들으시면 될 것 같아요.

가장 기본적은 SR은 <Jelinek et al 1976, Continuous Speech Recognition by Statistical Methods>에서 제시되었습니다. Word $$W$$가 있을 때, 이를 speaker가 말해서 오디오 $$A$$를 만들고, 이를 ASR 모델로 다시 $$W'$$ 를 복원하는 방식입니다. 이 때 $$W'$$은 다음과 같은 방식으로 구합니다.

$$
\hat{W} = \arg\max_Wp(W|A) = \arg\max_W p(A|W)p(W) = \arg\max_W \{\ln p(A|W) + \ln p(W)\}
$$

$$
\hat{W}= \arg\max \{\lambda \ln p(A|W) + \ln p(W)\}
$$

$$
$$
이 때, $$\lambda \ln p(A|W)$$ 은 Acoustic model이고, $$ \ln p(W)$$은 Language model입니다.

그 다음 단계로 넘어가기 전에, Language model을 다시 복습해보면, 어떤 word sequence의 등장 확률은 처음 단어가 들어갈 확률에다가 그전의 단어들이 주어졌을 때 다음 해당 단어가 등장할 조건부 확률을 계산하는 방식이지만, 편의상 n-gram으로 sequence를 잘라서, 그 전 n개의 단어들만 보고 다음 단어가 무엇이 나올 지를 예측합니다. 그 다음 Acoustic Model은 Hidden Markov Model이 기본이었습니다. 1990년대에 들어와서 SR에 인공신경망을 적용하려는 시도가 나타났는데요, 잘 안됬습니다 (ㅋㅋ...)


# 1. CD-GMM-HMM
1970년대에 처음 ASR이 제안되기 시작해서, 2010년도에 DBN이 적용되기 전까지 가장 좋은 성능을 보여주었던 모델입니다. 구조 자체가 뒤에서도 쓰이는 경우가 많으므로 뒤에서 다루고 이건 넘어가겠습니다.


# 2. CD-DNN-HMM
처음으로 딥러닝을 써서 기존의 모델보다 훨씬 좋은 성능을 보여준 모델입니다.

>  [Context-Dependent Pre-Trained Deep Neural Networks for Large-Vocabulary Speech Recognition](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/dbn4lvcsr-transaslp.pdf) 

Input으로 senones(tied triphone states)을 받습니다. 기존 CD-GMM-HMM에서 HMM을 가져와, 이를 바탕으로 정답에 대한 HMM input을 역으로 만든 다음에, 이를 DNN의 training label로 사용합니다. 그 다음 아래의 MLP만 학습합니다. 5개 정도 쌓았을 때 70%의 sentence accuracy를 보여줍니다. 특히, 데이터가 많아짐에 따라 계속 성능이 좋아지는 것을 확인할 수 있었습니다.

![Imgur](https://i.imgur.com/oHifiX5.png)



# 3. Small Variations
이 후에, 이 구조를 바탕으로 다양한 시도들이 있었는데요, 각각의 결과는 다루지 않고 간단하게 어떤 시도들이 있었는 지를 살펴보겠습니다.

- 대부분의 모델에서 뒷단의 HMM 구조는 그대로 두었습니다.
- Cross entropy loss를 Sequence level MMI로 바꾸었습니다. (상대적 20% 상승)
- Feature를 Mel-Frequency Cepstral Coefficients(MFCC)에서 FilterBanks로 바꾸었습니다. (상대적 5% 상승)
- Batch Normalization, distributed SGD, dropout을 사용해서 빠른 수렴과 약간의 성능 향상이 있었습니다.
- Acoustic Model을 DNN에서 CNN / CTC / CLDNN 으로 바꿔서 약간의 향상이 있었습니다.
- Language Model을 HMM에서 RNN으로 바꿔서 시도해봤습니다.
- Data를 양적으로도 추가하고, noisy, accent 등의 다양한 데이터도 추가했습니다.



# 4. Siri Architecture
다음으로는, 우리가 가장 많이 쓰는 ASR 모델인 siri의 모델을 소개하고 있네요. 아무래도 제품단으로 가져가려다 보니 경량화까지 같이 고려한 흔적이 보입니다.

> https://machinelearning.apple.com/2017/10/01/hey-siri.html

![Hey Siri DNN](https://imgur.com/TENsgM2.png)

Siri는 음성을 받으면 이를 ASR에 태워서 text로 만든 다음, 이를 바탕으로 NLU를 통해 intent를 뽑습니다. 이를 바탕으로 action을 취합니다. 여기서 siri가 사용한 구조는 약 0.2초 정도를 하나의 윈도우로 하고, acoustic model에는 5-layer MLP를(각각은 다 같은 사이즈로 상황에 따라 32 / 128 / 192 중 선택) 사용했고, 두 개의 네트워크를 통해 첫번째 네트워크는 초기 인식, 두번째 네트워크는 재확인용으로 사용했다고 합니다. 첫번째 네트워크는 더 작은 units를 사용했습니다(첫번째는 32, 두번째는 192). 각각의 네트워크에서 DNN의 output을 가지고 HMM scorer를 통해 text로 만들었습니다.

알고보니, 첫번째 네트워크가 작은 이유가, 24시간 내내 켜있어야 사용자의 말을 들을 수 있어서, 배터리 효율을 위해 작은 네트워크를 사용한 거였네요.



# 마치며

2년 전 내용이어서 그런지, 아님 입문 수업이어서 그런지 역사는 간단히 볼 수 있어서 좋았는데 최신 모델은 잘 안나왔네요. End-to-End Learning이나 최근 구조는 추가로 읽어봐야 할 것 같습니다.

