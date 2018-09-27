---
title: "FastText & GloVe"
excerpt: FastText와 GloVe 논문을 읽고 정리했습니다.
date: 2018-03-09
layout: post
category:
- [Natural Language Processing, Embedding]
- Paper Review
tag:
- Word Embedding
- FastText
- GloVe
---

# Enriching Word Vectors with Subword Information 

기본적으로 word단위로 끊어서 이를 embedding하는 Word2Vec과 달리, word를 character단위로 n-gram한 뒤, 이들의 **subword**를 embedding했더니 더 나은 성능을 보여주었다는 논문입니다.

### 1. Introduction

-	기존 모델은 각각의 단어를 벡터로 embedding
-	이는 내부 문장 구조를 무시(큰 한계), rich한 언어에 더 큰 한계
-	그래서 우리는 characer n-grams을 embedding함

### 3. Model

1.	General Model

	-	skip-gram model

		skip-gram model은 주어진 단어 시퀸스 $w*1,...,w_T$에 대해 log-likelihood를 최대화하는 목적을 가지고 있습니다. $$\sum*{t=1}^{T}\sum_{c \in C_t} \log p(w_c|w_t) $$ 여기서 $C_t$는 해당 단어 $w_t$근처에 있는 단어들의 집합입니다.

2.	Subword Model

	skip-gram 모델은 단어 자체의 internal structure를 고려하지 않는다는 단점이 있습니다. 따라서, 이를 해결하기 위해 다른 scoring fuction을 제안합니다. > Notations

	-	$G_w$ : 단어 $w$에 속하는 n-grams subwords
	-	$z_g$ : 각각의 $g$($g \in G_w$)의 벡터 표현

$$ s(w,c)=\sum_{g \in G_w} z_g^T v_c $$

### 4. Experimental Setup

1.	Baseline

	-	c의 word2vec에 있는 skipgram과 cbow 모델로 비교

2.	Optimization

	-	stochastic gradient descent on the negative log likelihood > **Stochastic gradient descent?** > > Gradient descent를 할 때 전체 데이터에 대해 optimization을 하지 않고 mini-batch에 대해 계산하여 optimization을 진행. 정확도는 다소 떨어지겠지만 계산 속도가 많이 빠르기 때문에 이를 여러번 실행하면 더 효율적으로 최적화 가능

	> **Negative log-likelihood?**
	>
	> 보통은 Maximum log-likelihood를 사용. 입력값 $X$와 파라메티 $\theta$가 주어졌을 때 정답 $Y$가 나타날 확률인 likelihood $P(Y|X;\theta)$를 최대화했었음. 정보이론으로 접근하면 두 확률분포 $p$, $q$ 사이의 차이를 계산하는 데에는 cross entropy가 사용되는데, 이 때 cross entropy가 파라메터 $\theta$안에서의 negative log-likelihood의 기댓값입니다. 그냥 cross entropy로 계산할 수도 있지만($H(P,Q)=-\sum_x P(x)\log Q(x)$) 손실함수로 negative log-likelihood를 사용할 경우, 1)만들려는 모델에 대한 다양한 확률분포를 가정할 수 있게 되 유연하게 대응가능하고, 2)

### 5. Results

1.	Human similarity judgement

	사람의 유사도 평가와, 실제 단어 embedding의 유사도의 correlation 비교 결과 짱

2.	Word analogy tasks

	$A:B = C:D$이다. 같은 문제에서 $D$를 예측하는 문제를 풀어봄.

3.	Comparison with morphological representations

	word similarity 문제에 적용함

4.	Effect of size of the training data

	약 20%의 데이터만 가지고 전체 데이터의 embedding을 잘 해결해주었다. 대박

5.	Effect of the size of n-grams

	3-6이 좋다.

### 6. Qualitative Analysis

1.	Nearest neighbors : baseline보다 좋다
2.	Character n-grams and morphemes :

### 7. Conclusion

-	character 단위의 n-grams로 이루어진 subword를 활용해서 embedding
-	train 빠름, preprocessing 필요없음
-	baseline 보다 훨씬 좋은 성능을 보임
-	open-source화함 ---

# Global Vectors for Word Representations (GloVe)
word-word co-occurrence matrix에서 nonzero인 요소들만 학습하여 좋은 성능을 보인 word embedding 방법입니다. word-context의 sparse한 matrix에서 SVD를 통해 차원축소를 해서 좋은 결과를 보인 LSA보다 좋은 효과를 보여준다고 합니다.

### 1. Introduction

기존에 word vector와 관련된 논문들은 크게 1) global matrix factorization, 2) local context window methods 정도로 나눌 수 있습니다. 1의 대표적인 방법인 LSA는 효율적으로 통계적 정보를 반영하지만, word analogy task($A:B=C:?$)같은 문제는 잘 풀지 못합니다. 2는 analogy 같은 문제들은 잘풀지만, 극소적으로 학습하기 때문에 corpus의 통계량을 반영하지 못합니다. 여기서는, specific weighted least squares 모델을 제안하여 전체 데이터셋에서의 word-word co-occurence matrix를 통해 효율적으로 통계량을 활용합니다. 이 모델은 word analogy에서 75%의 정확도를 보이고 있습니다.

### 3. The GloVe Model

단어의 출연도에 대한 통계량은 모든 word representation 관련 비지도 학습에서 사용됩니다. 물론 어떤 원리로 얼마나 의미를 반영하는지는 아직 숙제가 남아있습니다. 이 숙제를 해결하기 위해서 이 논문에서 GloVe를 제안한 것인데, 쉽게 표현하면 global corpus statistics(co-occurence matrix)를 활용한 word representation 방식입니다.

간단한 예시를 통해 어떻게 co-occurence matrix에서 의미를 도출할 수 있는지 살펴보겠습니다. Co-occurence matrix를 계산하면 어떤 단어들이 밀접하게 관련이 있고 어떤 단어들이 관련이 없는지 더 직관적으로 알 수 있습니다.

### 5. Conclusion

요즘 빈도수 기반 word representation이 나을지 prediction-based word representation이 나을지 논쟁이 있어왔다. 이 모델은 prediction-based model에 counts-based information(co-occurence matrix)를 넣어 좋은 결과를 보여주었다.
