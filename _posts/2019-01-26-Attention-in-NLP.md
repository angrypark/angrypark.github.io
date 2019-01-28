---
title: "Attention in NLP"
excerpt: 이 글에서는 attention이 무엇인지, 몇 개의 중요한 논문들을 중심으로 정리하고 NLP에서 어떻게 쓰이는 지를 정리해보았습니다.
date: 2019-01-26
layout: post
category:

- [Natural Language Processing]

tag:

- Attention

---

이 글에서는 attention이 무엇인지, 몇 개의 중요한 논문들을 중심으로 정리하고 NLP에서 어떻게 쓰이는 지를 정리해보았습니다.


> **목차**
- [기존 Encoder-Decoder 구조에서 생기는 문제](#기존-encoder-decoder-구조에서-생기는-문제)
- [Basic Idea](#basic-idea)
- [Attention Score Functions](#attention-score-functions)
- [What Do We Attend To?](#what-do-we-attend-to)
- [Multi-headed Attention](#multi-headed-attention)
- [Transformer](#transformer)



# 기존 Encoder-Decoder 구조에서 생기는 문제

Encoder-Decoder 구조에서 가장 중요한 부분은 input sequence를 어떻게 vector화할 것이냐는 문제입니다. 특히 NLP에서는 input sequence이가 dynamic할 구조일 때가 많으므로, 이를 고정된 길이의 벡터로 만들면서 문제가 발생하는 경우가 많습니다. 즉, "안녕" 이라는 문장이나 "오늘 날씨는 좋던데 미세먼지는 심하니깐 나갈 때 마스크 꼭 쓰고 나가렴!" 이라는 문장이 담고 있는 정보의 양이 매우 다름에도 encoder-decoder구조에서는 같은 길이의 vector로 바꿔야 하죠. Attention은 그 단어에서 알 수 있는 것처럼, sequence data에서 상황에 따라 어느 부분에 특히 더 주목을 해야하는 지를 반영함으로써 정보 손실도 줄이고 더 직관적으로 문제를 해결하기 위해 처음 제안되었습니다.

# Basic Idea (Bahdanau Attention)

> 논문 : [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)

가장 기본적인 아이디어는 encode할 때는 각각의 단어를 vector로 만들고, 각각을 attention weight에 따라 weighted sum을 한 다음, 이를 활용하여 다음 단어가 무엇일 지를 선택하는 것입니다. 

논문은 이 방식을 NMT에 사용하였는데요, bidirectional RNN을 encoder로 사용하고, $$i$$번째 단어에 대해 모든 단어에 대한 encoder output을 합쳐서 context vector로 만드는데, 이 때 단순 sum이 아닌 weight $$\alpha_{ij}$$​를 곱해서 weighted sum을 한 것입니다(아래 첫번째 수식). 이 때 $$i$$번째 단어에 대한 $$j$$번째 단어의 attention weight는 아래 수식 처럼 $$i$$번째 단어와 $$j$$번째의 원래 encoder output끼리를 feedforward neural network(attention weight를 만드는 모델을 논문에서는 align 모델이라고 부릅니다)를 태워서 만듭니다(아래 두번째 수식).

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
$$

$$
e_{ij} = a(s_{i-1}, h_j)
$$

align 모델을 Multi-layer Perceptron으로 만든 이유는 비선형성을 반영하고자 한 것이라고 하구요, 결국 이 align 모델은 NMT에서 같은 의미를 가진 단어를 잘 정렬하고(그래서 align) 짝지어 주기 위해서 있는 겁니다. NMT에서의 cost function 자체를 loss로 backpropagation 했구요.

# Attention Score Functions

위 논문 이후로 이 attention score $$\alpha_{ij}$$ 를 어떻게 만들 지에 대한 몇가지 변형들이 생겼는데요, 이를 정리해보겠습니다. 단어를 통일하기 위해 만들고자 하는 decoder state를 $$q$$ (query vector), 여기에 쓰이는 모든 encoder states를 $$k$$ (key vector)라고 하겠습니다(이는 뒤에서 다룰 Attention is All You Need 논문에서 나온 정의입니다). 이 단어를 이용한다면 $$\alpha_{ij}$$ 는 $$i$$ 번째의 query vector를 만들기 위한 $$i-j$$ key vector들 사이의 attention score라고 할 수 있겠죠.

## (1) Multi-layer Perceptron (Bahdanau et al. 2015)

$$
a(q, k) = w_2^T \tanh (W_1[q;k])
$$

위 논문의 MLP를 다시 적은 건데요, 이 방법은 나름 유연하고 큰 데이터에 활용하기 좋다는 장점이 있습니다. 

## (2) Bilinear (Luong et al. 2015)

$$
a(q, k) = q^TWk
$$

같은 연도에 나온 Lunong Attention은 $$q$$ 와 $$k$$ 사이에 weight matrix $$W$$ 하나를 곱해서 만들어줍니다.

## (3) Dot Product (Luong et al. 2015)

$$
a(q, k) = q^Tk
$$

2와 유사하지만, 그냥 $$q$$ 와 $$k$$ 를 dot product해서 이를 attention으로 쓰는 방법도 제안되었습니다. 이는 아예 학습시킬 파라미터가 없기 때문에 좋지만, $$q$$ 와 $$k$$ 의 길이를 같게 해야 한다는 단점이 있습니다.

## (4) Scaled Dot Product (Vaswani et al. 2017)

$$
a(q, k) = \frac{q^Tk}{\sqrt{\mid{k}\mid}}
$$

최근에 나온 논문 중에서 3을 개선 시킨 논문인데요. 기본적인 접근은 dot product 결과가 $$q$$ 와 $$k$$ 의 차원에 비례하여 증가하므로, 이를 벡터의 크기로 나눠주는 겁니다. 

# What Do We Attend To?

지금까지의 방법론들은 다 input sentence의 RNN output에다가 attention을 써서 이를 decoding에 활용했습니다. 이제 좀더 다양한 방식으로 attention을 맥이는 방법을 알아보겠습니다.

## (1) Input Sentence 

가장 기본적인 방법으로 그 전/ 그 후 input sentence들에다가 attention을 주는 방법입니다.

### - Copying Mechanism (Gu et al. 2016)

> 논문 : [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/pdf/1603.06393)

이 방법은 output sequence에 input sequences의 단어들이 자주 중복될 때, 이를 잘 copy하기 위해 처음 제안되었습니다. 예를 들어 대화를 이끌어 나갈 때, 기존에 나왔던 단어들을 활용해서 대답해야 하는 경우가 많죠. 

![copynet](/images/2019-01/190126_copynet.png)

### - Lexicon Bias (Arthur et al. 2016)

위의 copying mechanism과 유사한데, 하나의 dictionary를 구축하고 있다가 이를 활용하여 generate하는 방식입니다.

## (2) Previously Generated Things

그 다음은 그 전 generated된 결과에 attention을 주는 방식입니다. Language Modeling에서 중요한 것은, 말하고 있는 문장들이 일관된 흐름으로 이어져야 한다는 점입니다. 이를 반영하기 위해 예전에 generated된 단어들에다가 attention을 맥여서 그걸로 계속 generate하는 방식으로 쓰일 수 있습니다(Merity et al. 2016). Translation에서도 유사하게 input과 그 전 input에 attention을 매깁니다(Vaswani et al. 2017).

그런데 이 방법론의 문제점은, 한번 error가 생성되기 시작되면 그 error자체가 attention을 통해 계속 연결된다는 점입니다. 

## (3) Hierachical Structures (Yang et al. 2016)

> 논문 : [Hierachical Attention Networks for Document Classification](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)

이 방법은 document level representation을 만들 때 쓰일 수 있습니다. 쉽게 생각하면, sentence representation을 만들 때 word들에 대해 attention을 맥이고, document representation을 만들 때 sentence들에 대해 attention을 맥이는 겁니다. 이는 document classification같이 Input sequence가 매우 긴 상황에서 사용해볼 수 있습니다. 

## (4) Multiple Sources 

이제 문장들이 아니라 더 근본적인 출처를 다양화하는 방법을 알아보겠습니다.

### - Multiple Sentences (Zoph et al. 2015)

NMT에서 사용된 방법인데요, 하나의 target sentence를 생성해낼 때, 여러 개의 source 문장들을 동시에 사용해서 generate하는 방식입니다. 즉, 한국어를 만들 때 영어와 일본어 데이터를 동시에 써서 만드는 것이죠.

### - Multiple Stratigies (Huang et al. 2016)

이 논문에서는 input image의 CNN output에다가 attention을 맥여서 sentence representation을 만들어냅니다.

## (5) Self Attention (Cheng et al. 2016)

그 다음은 self attention입니다. 처음에 제안된건 2016년도 인데요, 하나의 문장에서 각각의 단어가 같은 문장의 다른 단어에 연관되어 있다는 관점에서 시작되었습니다. 즉 context sensitive한 encoding을 만들고자 한 것이죠.

![self_attention](/images/2019-01/190126_self_attention.png)

이 방법의 장점은 같은 문장끼리 attention을 구하기 때문에 길이가 항상 같다는 점입니다. 

# Multi-headed Attention

기존의 방법들은 attention을 주고자 하는 구조에다가 attention을 각각 하나의 vector로 표현하고자 했던 것에 비해, multi-headed attention은 기본적으로 하나의 attention이 문장의 여러 부분에 영향을 줍니다. 

![multi_head_attention](/images/2019-01/190126_multi_head_attention.png)

 먼저 Allamanis et al 2016에서는 copy하는 과정과 regular 과정에 각각을 attention을 주게 됩니다. 'copy'하는 과정이 이전에 어느 부분을 가져다 쓸지에 주목한다면, 'regular' attention은 다음 결정을 만들기 위해 어느 부분에 집중할 것인지를 정하는 것이죠.



# Transformer

> 논문 : [Attention is All You Need](https://arxiv.org/abs/1706.03762)

마지막으로 transformer에 대해 알아보겠습니다. 많은 사람들이 self attention = transformer 라고 하는데요, 엄밀히 따지면 다릅니다. 다만 이 논문이 2016년도에 제안된 self attention을 많이 참고한 것은 사실입니다. 이 transformer는 크게 3가지 특징이 있습니다.

- Attention **만을** 사용한 sequence to sequence 모델입니다.

- WMT에서 압도적인 성능을 보여줬습니다.
- 단순 matrix 곱으로만 되어있기 때문에 빠릅니다.

일단 구조를 그림으로 보면 다음과 같습니다.

![transformer](/images/2019-01/190126_transformer.png)

안에를 자세히 보면 이전에 언급했었던 attention 방식들이 많이 들어가 있습니다. 크게 4가지 attention trick이 있는데 다음과 같습니다.

- Self Attention : 각각의 layer는 하나의 단어를 같은 문장 내 다른 단어들과 연결합니다.
- Multi-headed Attention :  8개의 attention head가 따로 학습됩니다.
- Normalized Dot-product Attention : Dot product에서 나오는 bias를 제거해줍니다.
- Positional Encodings : RNN이 없어도 문장 내에서의 위치를 잊어버리지 않도록 position을 같이 encoding 해줍니다.

쉽게 설명하면 self attention을 multi-headed로 구현하고, 그 값에다가 feed forward network를 태워서 비선형성을 준 구조입니다. 그 밖에도 residual connection을 통해 gradient가 사라지는 것을 방지한 트릭도 있습니다.

학습할 때의 트릭은 다음과 같습니다.

- Layer Normalization : Layer 각각의 값들이 정상적인 범위 안에 있도록 normalize해줍니다.
- Specialized Training Schedule : Adam optimizer의 default값을 개선시켜서 learning rate scheduling을 진행하였습니다. warm-start, dropping등이 여기서 쓰였죠

- Label Smoothing : traing 과정에 어느 정도의 uncertainty를 집어넣었습니다.



지금까지 다양한 attention 기법들이 어떻게 생겨났고 각각은 대략적으로 어떻게 쓰이는 지를 정리해보았습니다. 마지막 transformer의 경우, 워낙 중요한 논문이고 이 만을 정리한 좋은 블로그 포스트가 있으므로 더 알고 싶으시면 꼭 읽어 보시길 바랍니다.

> [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)



# Reference

이 포스트는 CMU NLP 2018 강의 중 9강 Attention ([영상](https://www.youtube.com/watch?v=ullLRKZ99qQ&list=PL8PYTP1V4I8Ba7-rY4FoB4-jfuJ7VDKEE&index=21)) 을 기반으로, 관련 논문들을 읽어보며 정리하였습니다. 각각의 reference는 글 본문에 있습니다.

