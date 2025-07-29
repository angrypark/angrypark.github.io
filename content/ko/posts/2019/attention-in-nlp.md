+++
title = "Attention in NLP"
description = "이 글에서는 attention이 무엇인지, 몇 개의 중요한 논문들을 중심으로 정리하고 NLP에서 어떻게 쓰이는 지를 정리해보았습니다."
date = 2019-01-26T00:00:00+09:00
draft = false
tags = ["attention", "nlp", "transformer", "neural-network"]
categories = ["natural-language-processing"]
language = "ko"
url = "/ko/posts/2019/attention-in-nlp/"
+++

이 글에서는 attention이 무엇인지, 몇 개의 중요한 논문들을 중심으로 정리하고 NLP에서 어떻게 쓰이는 지를 정리해보았습니다.

> **목차**
- [기존 Encoder-Decoder 구조에서 생기는 문제](#기존-encoder-decoder-구조에서-생기는-문제)
- [Basic Idea](#basic-idea)
- [Attention Score Functions](#attention-score-functions)
- [What Do We Attend To?](#what-do-we-attend-to)
- [Multi-headed Attention](#multi-headed-attention)
- [Transformer](#transformer)

## 기존 Encoder-Decoder 구조에서 생기는 문제

Encoder-Decoder 구조에서 가장 중요한 부분은 input sequence를 어떻게 vector화할 것이냐는 문제입니다. 특히 NLP에서는 input sequence이가 dynamic할 구조일 때가 많으므로, 이를 고정된 길이의 벡터로 만들면서 문제가 발생하는 경우가 많습니다. 즉, "안녕" 이라는 문장이나 "오늘 날씨는 좋던데 미세먼지는 심하니깐 나갈 때 마스크 꼭 쓰고 나가렴!" 이라는 문장이 담고 있는 정보의 양이 매우 다름에도 encoder-decoder구조에서는 같은 길이의 vector로 바꿔야 하죠. Attention은 그 단어에서 알 수 있는 것처럼, sequence data에서 상황에 따라 어느 부분에 특히 더 주목을 해야하는 지를 반영함으로써 정보 손실도 줄이고 더 직관적으로 문제를 해결하기 위해 처음 제안되었습니다.

## Basic Idea (Bahdanau Attention)

> 논문 : [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)

가장 기본적인 아이디어는 encode할 때는 각각의 단어를 vector로 만들고, 각각을 attention weight에 따라 weighted sum을 한 다음, 이를 활용하여 다음 단어가 무엇일 지를 선택하는 것입니다. 

논문은 이 방식을 NMT에 사용하였는데요, bidirectional RNN을 encoder로 사용하고, $i$번째 단어에 대해 모든 단어에 대한 encoder output을 합쳐서 context vector로 만드는데, 이 때 단순 sum이 아닌 weight $\alpha_{ij}$를 곱해서 weighted sum을 한 것입니다(아래 첫번째 수식). 이 때 $i$번째 단어에 대한 $j$번째 단어의 attention weight는 아래 수식 처럼 $i$번째 단어와 $j$번째의 원래 encoder output끼리를 feedforward neural network(attention weight를 만드는 모델을 논문에서는 align 모델이라고 부릅니다)를 태워서 만듭니다(아래 두번째 수식).

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
$$

$$
e_{ij} = a(s_{i-1}, h_j)
$$

align 모델을 Multi-layer Perceptron으로 만든 이유는 비선형성을 반영하고자 한 것이라고 하구요, 결국 이 align 모델은 NMT에서 같은 의미를 가진 단어를 잘 정렬하고(그래서 align) 짝지어 주기 위해서 있는 겁니다. NMT에서의 cost function 자체를 loss로 backpropagation 했구요.

## Attention Score Functions

위 논문 이후로 이 attention score $\alpha_{ij}$를 어떻게 만들 지에 대한 몇가지 변형들이 생겼는데요, 이를 정리해보겠습니다. 단어를 통일하기 위해 만들고자 하는 decoder state를 $q$ (query vector), 여기에 쓰이는 모든 encoder states를 $k$ (key vector)라고 하겠습니다(이는 뒤에서 다룰 Attention is All You Need 논문에서 나온 정의입니다). 이 단어를 이용한다면 $\alpha_{ij}$는 $i$번째의 query vector를 만들기 위한 $i-j$ key vector들 사이의 attention score라고 할 수 있겠죠.

### (1) Multi-layer Perceptron (Bahdanau et al. 2015)

$$
a(q, k) = w_2^T \tanh (W_1[q;k])
$$

위 논문의 MLP를 다시 적은 건데요, 이 방법은 나름 유연하고 큰 데이터에 활용하기 좋다는 장점이 있습니다. 

### (2) Bilinear (Luong et al. 2015)

$$
a(q, k) = q^TWk
$$

같은 연도에 나온 Lunong Attention은 $q$와 $k$ 사이에 weight matrix $W$ 하나를 곱해서 만들어줍니다.

### (3) Dot Product (Luong et al. 2015)

$$
a(q, k) = q^Tk
$$

2와 유사하지만, 그냥 $q$와 $k$를 dot product해서 이를 attention으로 쓰는 방법도 제안되었습니다. 이는 아예 학습시킬 파라미터가 없기 때문에 좋지만, $q$와 $k$의 길이를 같게 해야 한다는 단점이 있습니다.

### (4) Scaled Dot Product (Vaswani et al. 2017)

$$
a(q, k) = \frac{q^Tk}{\sqrt{\mid{k}\mid}}
$$

최근에 나온 논문 중에서 3을 개선 시킨 논문인데요. 기본적인 접근은 dot product 결과가 $q$와 $k$의 차원에 비례하여 증가하므로, 이를 벡터의 크기로 나눠주는 겁니다. 

## What Do We Attend To?

지금까지의 방법론들은 다 input sentence의 RNN output에다가 attention을 써서 이를 decoding에 활용했습니다. 이제 좀더 다양한 방식으로 attention을 맥이는 방법을 알아보겠습니다.

### (1) Input Sentence 

가장 기본적인 방법으로 그 전/ 그 후 input sentence들에다가 attention을 주는 방법입니다.

#### - Copying Mechanism (Gu et al. 2016)

> 논문 : [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/pdf/1603.06393)

이 방법은 output sequence에 input sequences의 단어들이 자주 중복될 때, 이를 잘 copy하기 위해 처음 제안되었습니다. 예를 들어 대화를 이끌어 나갈 때, 기존에 나왔던 단어들을 활용해서 대답해야 하는 경우가 많죠.

*이 글은 원본의 일부만 포함하고 있습니다. 전체 내용은 이전 블로그에서 확인하실 수 있습니다.* 