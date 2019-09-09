---
title: "Learning Matching Models with Weak Supervision for Response Selection in Retrieval-based Chatbots"
excerpt: Retrieval 챗봇에서 지적되는 Label의 oversimplified 문제를 weak annotator로 해결하는 ACL 2018 논문입니다.
date: 2018-09-27
layout: post
category:
- [Natural Language Processing, Chatbots]
- Paper Review
tag:
- weak supervision
- ACL2018
- retrieval chatbots
- Seq2Seq
- matching model
---

## Retrieval Based Chatbots vs Generation Based Chatbots
챗봇을 만드는 방법은 크게 2가지로 나눌 수 있습니다. 답변의 후보군을 정한 다음, 해당 query에 알맞은 적절한 reply를 후보군에서 찾아서 내보내는 방법(Retrieval Based)과, 아예 모델이 답변을 만들어내는 방법(Generation Based)입니다. 전자는 classification으로 접근할 수 있고, 나오는 답변을 직접 handle할 수 있다는 장점이 있지만, 주어진 답변의 coverage와 domain에 성능이 매우 달라질 수 있고, 답변이 제한적이라는 단점이 있습니다. 후자는 전자에 비해 답변의 variation이 훨씬 크다는 장점이 있지만, 애초에 문맥적으로 옳지 않거나, 나오는 답변을 handle할 수 없다는 단점이 있습니다. Generation Based의 단점이 워낙 심각하기 때문에(문법 자체가 틀린 경우가 많고, 부적절한 답이 나올 수 있음) 아직까지는 deep learning 방법을 쓰더라도 retrieval based로 접근하는 경우가 많습니다.

## 기존 Retrieval Based Chatbots의 문제
기존에 알려진 문제(답변이 제한적이다, 답변 후보군에 따라 성능이 좌우된다) 이외에도 모델을 학습시키는 데에 있어서 발생하는 문제가 있습니다. 정상적인 대화를 어느정도 가지고 있을 때 이를 가지고 데이터를 만들 때, 보통 정상적인 대화의 각각의 pair를 1(positive), 나머지는 random negative sampling을 통해 0(negative)로 정의내린 다음 데이터를 만들지만, 이는 다음과 같은 문제를 가지고 있습니다.

#### - Label Imbalance
이론상으로 random negative는 positive data에 주어진 pair 수가 n일 때 n(n-1)개만큼 만들 수 있지만 잘 학습하려면 지나치게 많은 negative를 넣으면 안되고, 결국 데이터의 수는 positive pair에 절대적으로 비례하는데 이는 비쌉니다.

#### - Weak negative vs Hard Negative
각각의 negative는 보통 학습할 때 batch에서 1~10개씩 뽑거나 전체 데이터에서 1~10개씩 뽑는데, 무작위로 뽑기 때문에 해당 negative가 모델이 학습하기 쉬운 negative일 확률도 있고, 반대로 random하게 뽑았는데 해당 query에 말이되는 reply(positive)이지만 negative로 잘못 학습하는 경우도 있습니다. 더 큰 문제는 이런 다양한 경우를 모두 같은 0으로 학습하기 때문에 제대로 학습하지 못하거나, 학습되더라도 뻔한 수준에 머무를 가능성이 있다는 것입니다.

## Training Method
이 문제를 해결하기 위해서 이 논문에서는 기존에 1/0으로 학습하는 방식이 아닌 weak annotator 모델로 예측한 예측값을 사용하여 학습합니다.

$$
\sum_{i=1}^{N}\sum_{j=1}^{n}[r_{i, j}\log(\mathcal{M}(x_i, y_{i, j})) + (1-r_{i, j})\log({1-\mathcal{M}(x_i, y_{i, j})})]
$$

수식을 보면, 기존 방식은 positive pair의 matching score가 높을 수록 좋고, negative pair의 matching score가 낮을수록 좋게끔 선정한 것에 비해

$$
arg\min\sum_{i=1}^{N}\sum_{j=1}^{n}\max(0, \mathcal{M}(x_i, y_{i, j})-\mathcal{M}(x_i, y_{i, 1}) + s_{i, j}')
$$

여기서 제안한 방식은 negative일 경우 negative의 matching score(모델이 예측한 값)에서 positive의 matching score(모델이 예측한 값)을 빼고 거기에 weak annotator의 예측값을 더한게 작을수록 좋게끔 하는 것입니다. positive일 경우에는 $$\max(0, s_{i, j}')$$로 상수가 됩니다. 이 수식에서 특징은
- 0보다 작을 경우 0으로 clipping
- $$x_i$$에 bias가 생기지 않도록 normalize ($$s_{i, j}' = \max(0, \frac{s_{i, j}}{s_{i, 1}})$$)

입니다. 기본적인 framework는 annotator에 어떤 모델을 사용해도 되지만 여기서는 대용량의 사람 사이의 대화로 pre-trained된 Seq2Seq(+Attention)을 사용하였습니다.

위의 학습방법은 positive pair에 대한 matching score와 negative의 matching score의 차이를 최대한 크게 벌리는 데에는 기존과 같지만, $$s_{i, j}'$$를 통해 각각의 sample 간의 차이를 준다는 장점이 있다고 생각합니다. 한 예로 semantic하게 유사하지도 않은 negative에게는 더 낮은 점수를 주고, semantic하게는 유사할 경우에는 어느 정도의 점수를 주고 학습을 할 수 있습니다. 논문에서는 이를 
> *"An advantage of out method is that it turns the hard zero-one labels in the existing learning paradigm to soft matching scores"* 

라고 표현합니다.

## Weak Annotator
 Weak Annotator은 Adversarial한 두 개가 있다는 점에서 GAN과 비슷하지만, discriminator를 통해 generator를 더 잘 학습시키는 GAN과 달리 이 방법은 generator(weak annotator)를 통해 discriminator(classifier)를 더 잘 학습시킵니다. 기존에도 [Dehghani et al., 2017](https://arxiv.org/pdf/1711.11383)와 같이 supervisor를 쓴 경우는 있었지만 이 모델은 unsupervised supervisor라는 점이 의미가 있는 것 같습니다.

## Updating Seq2Seq model
 가장 재밌게 읽었던 부분이 이 부분인데요, 보통의 Seq2Seq 모델에서 발생하는 "safe response" 문제(많이 나오는 답변에 편향되는 문제)를 해결하기 위해 20 배치마다 policy gradient를(!!) 통해 update를 했다고 하는데, 결과적으로는 성능이 안좋았다고 합니다. 그 이유로
 - PG로 Seq2Seq를 발전시키는 방법 자체가 어렵다
 - "safe response"를 해결하는 것이 weak annotator의 성능을 높이는데 도움을 준다고 판단할 수 없다
를 들었습니다.

## Results
STC Dataset(Short Text Conversation, 1-turn)과 Douban Conversation Corpus(multi-turn)에 적용한 결과는 다음과 같습니다. 이 때 `num_negative_samples`는 20으로 했다고 합니다. (soft한 score로 인해 더 많은 negative를 넣을 수 있는 것도 장점인 것 같습니다.)

<img src="/images/2018-09-27/STC_result.png" alt="STC_result" width="350"/> <img src="/images/2018-09-27/DCC_result.png" alt="DCC_result" width="300"/>