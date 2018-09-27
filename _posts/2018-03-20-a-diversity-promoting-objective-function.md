---
title: "A Diversity-Promoting Objective Function for Neural Conversation Models"
excerpt: Seq-to-Seq model에 Maximum Mutual Information을 목적함수로 넣었더니 더 좋은 결과(more diverse, interesting, and appropriate responses)가 나왔다라는 논문입니다.
date: 2018-03-20
layout: post
category:
- Natural Language Processing
- Paper Review
tag:
- machine translation
- Seq2Seq
---

### 1. Introduction
- 기존 SMT(Statistical Machine Translation) 모델 소개 : 여러 문제점을 가지고 있음
- 문제점들 중 scalability와 language independence를 해결한 것이 Seq2Seq
- Seq2Seq도 답변이 너무 뻔함 (ex. "I don't know.", "I'm OK")
- 그런데 K-best list를 뽑아보면, 상황에 적합한 문장이 있긴 있는데 뻔한 답변보다는 랭크가 낮아서 안나왔던 것
- 이를 MMI로 해결하겠다.

**2. Related Works**(SMT, Seq2Seq 등의 논문들 소개), **3. Sequence-to-Sequence Models**(LSTM 설명)은 패스

### 4. MMI Models
#### - 용어 정의
- $S$ : input message sequence
- $N_S$ : number of words in $S$
- $T$ : {$t_1, t_2, ..., t_{N_T},EOS$}, response to source sequence
- $t$ : a word token which is associated with a K dimensional distinct word embedding $e_t$
- $V$ : vocabulary size

#### - 기존 Objective Function : source S에 대한 T의 log-likelihood
$$
\hat{T} = argmax_T(\log_p(T|S))
$$

#### - MMI : S와 T의 pairwise mutual information을 최대화하게끔 선택됨
$$
\log{\frac{p(S,T)}{p(S) p(T)}}
$$

이를 이용한 목적함수는
$$
\hat{T} = argmax_T(\log_p(T|S)-\log_p(T))
$$

여기에 분모에 대해 페널티를 추가하면,
$$
\hat{T} = argmax_T(\log_p(T|S)-\lambda \log_p(T))
$$

베이즈 정리를 이용하여 바꾸면,
$$
\log{p(T)} = \log_p(T|S) + \log_p(S) - \log_p(S|T)
$$

- MMI-antiLM : $\log_p(T|S)-\lambda \log_p(T)$

    anti language model이라고 불립니다. high-frequency, generic responses뿐만 아니라 fluent해서 ungrammatical output이 나올 수 있는 것들도 줄입니다. 이론적으로는 $\lambda$가 1보다 작으면 비문이 생기는 것을 방지하지만, 실제로는 뒤의 항으로 인해 비문들이 선택된다고 합니다. 이를 해결하기 위해서 $p(T)$를 계산할 때 $g(i)$를 곱해줍니다. 여기서 $g(i)$는 일정한 threshold보다 높으면 1, 아니면 0을 선정하여 일정 길이가 넘어가는 index의 토큰들을 안곱해줍니다.
    

- MMI-bidi : $(1-\lambda)\log_p(T)+\lambda \log_p(S|T)$
    
    위의 공식을 바로 구현하기는 매우 어렵다고 합니다(두번째 항이 target generation을 끝내야 계산할 수 있기 때문이죠). 따라서 먼저 standard Seq2Seq로 계산한 뒤에, N-best list에서 두번째 항을 기준으로 재정렬을 하면 된다고 합니다!!!! 이건 이해하기 쉽군요.

---

    
    
    