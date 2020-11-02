---
title: "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations"
excerpt: RecSys 2019에서 Google이 공개한 Youtube 추천 관련 논문입니다.
date: 2020-10-31
layout: post
category:

- [Recommender System]

tag:

- Youtube recommender system
- Item recommendation
- Two tower model

---

 이 논문을 처음 알게 된 것은 저번달에 Google Brain에서 [Tensorflow Recommenders](https://github.com/tensorflow/recommenders) 라는 라이브러리를 공개하면서 입니다. Youtube라는 거대한 추천시스템을 운영하고 있는 구글이 추천 시스템 관련 코드를 공개한다고 해서 집중해서 보게 되었습니다. 전체적인 내용은 [Tensorflow Blog](https://blog.tensorflow.org/2020/09/introducing-tensorflow-recommenders.html)에 더 자세히 나와있으니 읽어보시기 바랍니다.

 TFRS(TensorFlow Recommeners)의 목표는 다음과 같습니다. 

- 추천 후보군을 빠르고 유연하게 빌드
- Item, User, Context 정보를 자유롭게 사용하는 구조
- 다양한 objective를 동시에 학습하는 multi-task 구조
- 학습된 모델은 TF Serving으로 효율적으로 서빙

 사실 코드 자체는 크게 다양한 내용들이 있지는 않았지만, 제일 인상 깊었던 것은 코드에서 기본 모델로 소개한 Two Tower Model이었습니다. 바로 User와 Item을 아예 독립적으로 학습시켜 마지막 단에서 dot product로만 click / unclick을 예측하는 것인데, 생각하면 생각할 수록 좋은 구조더라구요. 비록 학습하는 단에서 user tower와 item tower가 interact 못하기 때문에 엄청난 성능을 낼 지는 미지수였지만, 구조 자체가 input feature의 제약이 없어서 가능한 feature를 자유롭게 넣을 수 있었고, inference할 때는 user별 embedding, item별 embedding으로 가지고 있다가 dot product로만 similarity를 계산해서 serving할 수 있기 때문에 ANN(Approximate Nearest Neighbors) 라이브러리와의 호환성도 좋아 보였습니다. 

![tfrs](/images/2020-10/201031_tfrs.gif)

<center>(출처 : https://blog.tensorflow.org/2020/09/introducing-tensorflow-recommenders.html)</center>

<br/>

 또하나의 장점은 메타 정보를 넣을 수 있다라는 것인데, 추천 시스템에서 자주 만나는 문제가 cold start problem입니다. Item이던 User던 처음에 사용 기록이 없을 경우 메타 정보를 활용할 수밖에 없는데, 이를 범용적으로 잘 모델링하기가 어렵습니다. 하지만 위의 two tower 모델은 사용 기록이 없어도 User나 Item의 메타 정보를 넣고 나머지는 평균값 같은 걸 넣으면 바로 modeling이 되기 때문에, cold start 문제도 알아서 잘 해결할 수 있을 것 같았습니다. 이 말은 dynamic하게 아이템 풀이 바뀌는 구조에서 잘 쓰일 수 있다는 말이기도 하구요.

 여기까지 생각을 하면서 Two Tower Model 관련 논문을 더 자세히 읽고 정리해 보았습니다. 원본 논문 제목은 ["Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations"](https://dl.acm.org/doi/10.1145/3298689.3346996)  이고, Google Brain in Youtube에서 공개하였으며, 지금 Youtube 추천 시스템에서 candidate generation에서 사용하고 있는 모델입니다.



# 1. Concept

 유튜브에서는 크게 2가지 stage로 나뉘어서 추천이 이뤄지고 있습니다. 바로 candidate generation과 ranking model인데요. 어떤 유저에게 어떤 영상을 추천해준다고 했을 때 1) candidate generation을 통해 전체 아이템 중에서 추천 후보군으로 뽑을 만한 몇 백개의 아이템을 추리고, 그 몇 백개 중에서 최종 추천에 나갈 몇 개는 2) ranking model이 결정합니다. 이 때 candidate generation은 실제 원하는 아이템이 반드시 추천 후보군에 있어야 하므로 recall@k 가 중요하고, ranking model은 실제 원하는 아이템이 최상위에 랭크되어야 하므로 nDCG@k, HR@k 등이 중요합니다.

<p align="center">
  <img src="/images/2020-10/201031_youtube.png" alt="Sublime's custom image"/>
</p>

 또 실제 추천시스템을 만들어야 하는 입장에서 생각해 본다면, candidate generation model은 전체 아이템 중에서 몇백개를 추려야 하기 때문에 최고의 성능보다는 적당한 성능과 빠른 inference 속도가 더 중요합니다. Inference를 빠르게 하여 수백~수천만개의 아이템 중 몇백개를 추리는 방법은 item의 좋은 embedding vector를 뽑아서 user가 최근에 소비한 item과 유사한 아이템을 후보군으로 뽑는 것입니다. 

 어떤 item의 embedding을 뽑는 방식은 간단하게는 word2vec과 bag of words를 결합하여 item의 text embedding을 뽑을 수도 있고, 영상이나 그림의 경우 pretrained image model을 활용해 low dimension의 feature를 뽑을 수도 있습니다. 이 논문에서 제안된 two tower model은 meta정보와 다양한 feature를 넣어서 좋은 user / item embedding을 만든 다음에 dot distance 기반으로 nearest neighbor를 찾는 방식으로 사용됩니다.



# 2. Modeling Overview

그러면 이 논문에서 핵심이 되는 two tower 구조와, 이 논문은 데이터 단에서는 positive pair만 가지고 있다가 batch 내에서 negative sample을 뽑았는데 이를 어떻게 했는지 알아보도록 하겠습니다.



## Two-tower Model

 사실 two tower model은 dual encoder라는 말로 자연어처리 분야에서 먼저 유행되었습니다. 어떤 두 문장의 관계를 예측하고자 하는 과제에서, 문장 하나마다 encoder를 태우고 거기서 나온 sentence representation으로 문장의 관계를 분류합니다. 이 때 사용하는 encoder로는 RNN, Transformer부터 최근에는 pretrained BERT 구조까지도 사용되었습니다. 분류하고자하는 label이 무한히 많을 때, 이를 multi label classification으로 접근하지 않고 query와 label를 input으로 해서 적합한지 아닌지를 판단하는 binary classification으로 접근한다는 점이 이 구조의 핵심입니다.



## Batch Negative Sampling

 Two-tower 모델을 학습한다고 했을 때 필요한 데이터는 어떤 user가 item을 클릭했다라는 데이터와 어떤 user가 item을 봤지만 click하지 않았다라는 데이터입니다. 하지만 이 click하지 않았다라는 데이터를 수집할 때 발생할 수 있는 몇가지 어려움이 있는데, 다음과 같습니다.

- 애초에 서비스 환경에 따라 노출되었다라는 데이터를 구할 수 없을 수도 있습니다.

- 데이터 크기: 보통 click에 비해 impression(노출)의 수는 압도적으로 많습니다. 이를 저장하기 시작한다면 데이터의 크기가 지나치게 늘어납니다.
- Serving bias]: unclick은 전적으로 당시에 추천 로직이 무엇이었느냐에 따라서 데이터가 결정됩니다. 따라서 해당 추천 로직에 따라 negative의 data distribution이 매우 달라집니다.
- Hard negative: 노출이라는 것도 결국 기존 로직에서 topk로 추천이 나간 결과 중에 실패한 것들인데, 기존 로직에서 topk로 노출되었다라는 것 자체가 이미 어려운 negative입니다.



 이를 해결하기 위해서 click데이터 만으로 학습을 할 때 batch 내에서 임의로 negative를 뽑는 것이 batch negative sampling입니다. Batch 단위로 positive pair들이 들어올 때, 순서를 어긋나게 해서 해당 user에게 다른 item을 맵핑시킨 다음에 이를 negative라고 생각하는 것입니다. Two Tower 모델에서는 중복된 계산을 피하기 위해 마지막 dot product 직전에서 batch negative sampling을 진행할 수 있습니다.

<p align="center">
  <img src="/images/2020-10/201031_negative_sampling.png" alt="Sublime's custom image" width="600"/>
</p>



 위 그림에서 query embedding과 item embedding을 matmul 계산을 하면 label matrix가 되고 이때 $$(i,i)$$ 열만 positive이고 나머지는 다 negative가 됩니다. 모든 걸 다 negative로 쓰지는 않고 negative sampling에도 여러가지 기본 방식이 존재합니다. 대표적인 것들은 다음과 같습니다.

- random negative: $$B$$-1개 중에서 k개를 임의로 sampling 합니다.
- hard negative: $$B$$-1개 중에서 모델이 판단하기에 가장 어려워했던 pair (dot product 값이 가장 높은 것들)만 sampling합니다.
- semi-hard negative: $$B$$-1개 중에서 모델이 판단하기에 다소 어려워했던 pair (dot product 값이 특정 range에 있는 것들) 중에서 k개를 sampling합니다.



 이러한 negative sampling을 two-tower model 마지막 단에서 진행하게 된다면 반드시 성능이 좋다라는 보장은 없지만, 적어도 데이터의 크기가 몇십분의 1로 줄고, 기존과 계산하는 것은 크게 다르지 않기 때문에 학습 속도도 몇 십배 증가한다고 볼 수 있습니다.

 하지만 반드시 좋은 점만 있는 것은 아닌데요, 앞서 설명드렸다시피 batch negative sampling은 추천 말고도 다른 분야에서도 많이 사용되지만 추천에서 batch negative sampling을 할 때 발생하는 문제점이 있습니다. 바로 popular item인데요. 추천에서는 item의 등장 확률이 특정 인기있는 것들에 매우 치우쳐 있습니다. 그만큼 click이 많이 일어난다는 것이니깐 positive sample을 바라볼 때는 상관이 없지만, 문제는 negative sampling할 때 negative 후보군들에도 popular item이 너무 많다라는 거죠. 이를 item frequency bias라고 말합니다. 이 논문에서는 이를 해결하기 위해서 matmul한 logit에다가 각 아이템별 sampling probability를 estimate해서 빼줍니다. 그렇게 되면 popular item에 대한 loss는 알아서 줄어들게 됩니다.



# 3. Stream Frequency Estimation

 그러면 아이템별로 batch에서 sampling될 확률은 어떻게 구할 수 있을까요? 결국 해당 아이템의 인기도를 시간에 따라 빠르게 estimate하면 되는데 이는 생각보다 어렵습니다. 같은 아이템에 대해서도 시간에 따라 인기가 급증할 수도 있고, 반대로 갑자기 사라질 수도 있습니다. 논문에서는 이를 간단한 알고리즘으로 구했습니다.

<p align="center">
  <img src="/images/2020-10/201031_stream_frequency_estimation.png" alt="Sublime's custom image" width="600"/>
</p>



 위 수식에서 $$h(y)$$는 $$y$$라는 item id가 무한히 증가할 수 있기 때문에 이를 hash하는 함수이고, $$A[h(y)]$$는 해당 $$y$$가 최근 몇번째 batch 전에서 나왔냐는 정보입니다. 바로 이전 batch에서 나왔으면 1이고, 한번도 나오지 않았으면 $$t$$가 되겠죠. 즉 얼마나 최근에 sampling되었냐를 나타내는 정보입니다. 

 배치마다 등장하는 아이템들에 대해서, 얼마나 최근에 sampling되었냐라는 정보와 그 전의 저장해두었던 sampling probability를 가지고 현재 sampling probability를 계산합니다. 기존에 계산해두었던 sampling probability는 $$(1-\alpha)$$로 time decay 시키고, 최근성을 반영해 update하는 방식입니다. 이를 바탕으로 dynamic하게 아이템의 등장 확률이 바뀌는 상황에서 특정 아이템의 sampling probability를 계산합니다.

 여기까지가 논문의 주요 이론적 내용들이고, 이제부터는 실제 유튜브 추천에서 어떤 feature와 함께 어떻게 사용되었는지를 알아보겠습니다.



# 4. Youtube Neural Retrieval Models

## Features

[1. Concept](#1-concept) 에서 언급했던 것처럼 유튜브에서 two-tower model은 candidate generation에서 사용됩니다. 각각의 feature들은 다음과 같습니다.

- training label: 오로지 positive만 사용하되, 적게 시청한 click은 0, 다 시청한 click은 1로 학습합니다.
- Item features: video_id, channel_id 등을 사용하며, 각각은 embedding lookup을 통해 trainable한 dense feature로 변환합니다.
- User features: 최근 본 몇개의 video id들의 embedding들을 가져와 이를 평균냅니다.

여기서 공통된 점은 video_id들은 word embedding과 유사하게 trainable한 dense vector로 변환한다는 것입니다. 



<p align="center">
  <img src="/images/2020-10/201031_two_tower_model.png" alt="tt" width="900"/>
</p>

 이 밖에도 해당 item의 view, like등의 정보도 들어가있지만 정확히 어떻게 scaling했는 지는 밝혀지지 않았고, 다만 [16년도 유튜브 추천 논문](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/45530.pdf)을 참고하면 0~1로 scaling했다고 짐작할 수 있습니다.



## Sequential Training

 그렇다면 데이터는 어떤 크기로 어느 주기로 학습시켰을까요? 유튜브에서는 매일 1번, 하루치 데이터를 받아서 학습을 시킨다고 합니다. 특이한 점은 일반적인 딥러닝 학습과 달리 epoch을 1번만 진행하고, shuffle을 하지 않는다고 합니다. Shuffle을 하지 않는 이유에는 워낙 시간에 따라 data distribution이 달라지기 때문에 data distribution shift를 잡아내기 위해서라고 합니다.



## Indexing and Model Serving

 마지막으로 indexing과 serving입니다. 유튜브는 아이템 수가 워낙 많기 때문에 추천에 나갈 아이템을 어느정도 추린다고 합니다. 논문에서는 "a set of videos are selected from YouTube Corpus based on certain crierion"이라고만 언급되어 있어서 그냥 rule-based로 추천 후보군을 뽑는 것 같습니다. 그 후보군들에 대해서는 위에서 학습된 모델로 각 후보 item별 embedding을 뽑게 됩니다. 그 다음 index training을 하게 되는데, 이 부분에 대해서는 논문에서 아예 언급되지 않았지만, 결국 추천을 나갈 때는 dot distance 기반 nearest neighbor를 내보내기 때문에 이를 더 빠르게 하기 위해 ANN(Approximate Nearest Neighbor) 라이브러리를 사용할 것이고, 그를 위한 학습을 진행한다고 추측할 수 있습니다.





<p align="center">
  <img src="/images/2020-10/201031_index_pipeline.png" alt="201031_index_pipeline" width="600"/>
</p>



# 5. Results

 지금까지 Youtube 추천 시스템에서 Candidate Generation에 사용되고 있는 Two Tower 모델에 대해 알아보았습니다. Youtube에서는 위의 구조로 몇백개의 후보군을 뽑은 다음에 Multi-Gate Mixture Of Experts 모델로 reranking을 하고 있습니다. 이 모델의 가장 큰 장점은 다양한 feature를 유연하게 넣을 수 있는 구조라는 점과, 메타 정보를 활용하기 때문에 dynamic한 추천 환경에서도 잘 사용되며, inference시에는 linear한 시간복잡도로 가능하다는 것입니다. 

이 논문은 제가 PR12에서 발표한 [영상](https://youtu.be/FSDuo9ybv8s)도 있으니 참고하시기 바랍니다.



## Reference

- [Yi et al 2019, Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://research.google/pubs/pub48840/)
- [Covington et al 2016, Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/45530.pdf)
- https://github.com/tensorflow/recommenders
- https://blog.tensorflow.org/2020/09/introducing-tensorflow-recommenders.html