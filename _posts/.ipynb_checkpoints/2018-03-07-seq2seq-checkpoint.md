---
title: "Sequence to Sequence(Seq2Seq) Paper Review"
excerpt: Sequence에서 다른 Sequence로 연결하는 일반적인 end-to-end 접근 방법을 소개합니다.
date: 2018-03-07
layout: post
category:
- Natural Language Processing
- Paper Review
tag:
- Seq2Seq
---

Sequence에서 다른 Sequence로 연결하는 일반적인 end-to-end 접근 방법을 소개합니다. (여기서 sequence는 연관된 연속 데이터를 의미) LSTM을 활용하여 input sequence를 정해진 벡터로 mapping하고, 다른 LSTM을 활용하여 그 벡터를 target seqeunce(여기선 예로 다른 언어)로 mapping합니다. train하는 과정에서 source data의 단어의 순서를 바꾸는 과정이 LSTM의 성능을 엄청나게 올렸습니다. 

### 1. Introduction
딥러닝은 기존의 머신러닝이 풀기 힘들었던 여러 복잡한 문제들(음성 인식)에 뛰어난 성능을 보여줍니다. 큰 네트워크에 미리 학습된 파라미터들이 있다면 나중에 쉽게 응용 가능할 것입니다.
하지만 DNN은 input과 output이 정해진 차원의 vector여야 합니다. 그걸 구현하기 위해서 여기서는 하나의 LSTM이 input sequence를 timestamp와 함께 읽으면서 large fixed-dimensional vector로 표현하였고, 두번째 LSTM은 vector에서 output sequence를 추출합니다. 

### 2. The Model
sequence를 학습하기 위한 가장 간단한 방법은 input sequence를 하나의 RNN을 활용해서 고정된 크기의 벡터로 변환하는 것입니다. 하지만 long term dependencies 문제로 인해 RNN이 제대로 학습되지 않을 수 있습니다. 그러나 LSTM은 이를 해결합니다. 

LSTM의 목표는 input sequence의 크기와 ouput sequence의 크기가 다르더라도 output sequence의 conditional probability를 잘 추정하는 것입니다. 먼저 input sequence($x_1$,..., $x_T$)를 LSTM의 마지막 hidden state를 활용해 정해진 크기의 벡터로 변환합니다. 그 다음 $y_1$,....,$y_{T'}$를 LSTM-LM으로 추정합니다.
> **LSTM-LM?**

> Language Model로 이전 단어가 들어왔을 때 다음에 해당 단어가 나올 확률을 계산하여 가장 나올 확률이 높은 단어를 추가하는 모델입니다.

$$
p(y_1,...,y_{T'}|x_1,...,x_T) = \prod_{t=1}^{T'}p(y_t|v,y_1,...,y_{t-1})
$$

그러나!!!! 여기서 소개하는 모델은 바로 위의 모델과 다음에서 다릅니다.
1. 2개의 LSTM : input sequence를 위한 LSTM, output sequence를 위한 LSTM
2. deep LSTM : shallow LSTM 대신 4 레이어의 깊은 LSTM 사용
3. input sequence의 순서를 역순으로 함

### 3. Experiments
그 다음은 성능 비교였는데, 뭐 대부분의 논문들이 그러하듯이 좋았다라는 말만 반복합니다.
- 긴 문장에서도 잘된다
- BLEU Score가 높다

### 5. 결론 : 좋다!
---
