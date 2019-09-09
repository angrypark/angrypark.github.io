---
title: "Multi Armed Bandit"
excerpt: 최근 Recommendar System에 대해 공부하면서, Multi-armed bandit이라는 분야에 대해 공부할 필요가 있다고 생각하던 차에 [A Survey of Online Experiment Design with the Stochastic Multi-Armed Bandit](https://arxiv.org/abs/1510.00757)을 바탕으로 정리해보았습니다.
date: 2019-02-05
layout: post
category:

- [Recommendar System]

tag:

- Multi-armed bandit

---

최근 Recommendar System에 대해 공부하면서, Multi-armed bandit이라는 분야에 대해 공부할 필요가 있다고 생각하던 차에 [A Survey of Online Experiment Design with the Stochastic Multi-Armed Bandit](https://arxiv.org/abs/1510.00757)을 바탕으로 정리해보았습니다.


> **목차**
- [기존 Encoder-Decoder 구조에서 생기는 문제](#기존-encoder-decoder-구조에서-생기는-문제)
- [Basic Idea](#basic-idea)
- [Attention Score Functions](#attention-score-functions)
- [What Do We Attend To?](#what-do-we-attend-to)
- [Multi-headed Attention](#multi-headed-attention)
- [Transformer](#transformer)



# 1. Concept
Multi-armed Bandit(이하 MAB)라는 단어가 나오게 된 배경은 겜블링입니다. 어떤 사람이 주어진 시간안에, 수익 분포가 다 다른 N개의 슬롯머신을 통해 최대의 수익을 얻는 방법은 무엇일까요? 만약 제한된 시간에 N개의 슬롯머신들을 당겨서 수익을 얻을 수 있는 기회가 주어진다면, 일단은 어느 시간 동안은 어느 슬롯 머신이 돈을 많이 얻을 수 있는 지 탐색하는 과정이 있어야 할꺼고(이를 Exploration이라고 합니다), 그 다음에는 자신이 판단하기에 괜찮은 슬롯 머신을 돌리면서 수익을 극대화하는 과정이 필요합니다(이를 Exploitation이라고 합니다).

![concept](/images/2019-02/190205_concept.png)

Exploration을 많이 하면, 어떤 슬롯머신이 더 성공확률이 높은 것인 지를 더 잘 파악할 수 있지만, 그걸 찾기만 하다가 막상 수익을 많이 얻지 못한다는 단점이 있구요, exploitaion을 많이 하면 알려진 분포들 사이에서는 그나마 괜찮은 수익을 얻을 수 있겠지만, 더 좋은 슬롯머신을 찾아서 시도하지 못했다는 아쉬움이 생기겠죠. 이를 **exploration-exploitation tradeoff**라고 합니다.

MAB는 이런 exploration-exploitation tradeoff를 잘 조절해나가면서 빠른 판단과 좋은 결과를 내기 위한 실행을 결정합니다. 일단은 환경과 반응하면서 학습한다는 점과 decision making을 한다는 관점에서 강화학습의 한 종류라고 볼 수 있구요. 추천 시스템이나 주식 투자, 의료 실험 등에서 모두 사용되고 있습니다.

기존 Supervised Learning과 가장 다른 점은, 실시간으로 이루어지는 exploration & exploitation와 변수에 자원(시간, 시도 횟수 등)을 넣었다는 점입니다. 기존 Supervised learning은 하나의 문제가 정해져 있고, 그 문제에 해당하는 데이터를 수집한 다음, 예측하고자 하는 값을 예측하는 decision boundary를 찾게 됩니다. 하지만 주식 투자나 추천 시스템에서는 예측하고자 하는 값이 자주 바뀌어서 데이터를 모으고 학습하고 이를 통해 예측하는 과정이 지나치게 오래 걸릴 때가 많습니다. 따라서 제한된 자원 내에서 최선의 수익을 얻기 위한 방법론 중 하나가 Mulit Armed Bandit입니다.



# 2. MAB와 기존 통계 기반 모델들과의 차이점
MAB 실험 환경은 어떤 시도에 따른 결과를 즉각적으로 받을 수 있는 환경인 경우가 대부분입니다. 따라서 MAB 알고리즘에 대해 알기 전에 먼저 생각해야 할 것은, MAB 실험환경에서 어떻게 알고리즘을 평가하냐는 것입니다. 기존의 supervised learning이나 unsupervised learning은 확실한 loss function이 있고 이를 최소화하자는 목적이 있습니다. 그러나 MAB 실험 환경에서는 실제 온라인 환경으로 바로 평가하지 않는 한(사실 이것도 온전하다고는 볼 수 없죠.) 해당 MAB 알고리즘이 어떤 성능을 가지고 있는 지에 대한 평가가 어렵습니다. 이를 측정하기 위해 regret, variance and bounds of regret, stationary, feedback delay들을 통해 MAB 알고리즘을 평가합니다.

## 1) Regret
Regret은 사전적 의미 그대로 이해하면 더 쉽습니다. 내가 선택을 했을 때, 나중에 결과를 확인하고 얼마나 후회할 것이냐입니다.
> *"The remorse(losses) felt after the fact as a result of dissatisfaction with the agent's (prior) choices."*

이를 해석하면 사전에 기대했던 결과와 실제 결과의 차이라고 볼 수 있구요, 해당 bandits 중 가장 optimal한 결과와 내 결과의 차이로도 볼 수 있습니다. 

$$
\bar{R}^E = \sum^{H}_{t=1}(\max_{i=1,2, ..., K}E[x_i,t ]) - \sum^{H}_{t=1} E[x_{S_t, t}]
$$

Regret에도 다양한 종류가 있지만, 위 수식은 선택된 arm에서의 기댓값과, 전체 arm에서의 가장 높은 기댓값과의 차이를 regret으로 정의내렸습니다. 즉, 이론적으로 사전에 각 arm별 분포를 정의내릴 수 있는데, 사전에 정의된 분포에 따른 기댓값의 최댓값과 MAB 알고리즘이 선택한 arm의 기댓값과의 차이를 구하는 것이죠. 이는 매우 직관적이고 쉽게 regret을 구할 수 있다는 장점이 있지만, 실제 서비스에 적용할 때 각 arm의 분포가 이론적으로 정의내린 분포와 다르면 결과가 매우 달라지기 쉽다는 단점이 있습니다.

## 2) Variance and Bounds of Regret
1)에서 언급한 regret은 결국 미리 정해놓은 분포(혹은 실제 분포)와 알고리즘이 예측한 분포가 얼마나 다른 지를 평가하는 지표입니다. 이를 supervised learning과 연결시켜 생각해보면, 위의 regret은 loss function의 일종이라고 볼 수 있습니다. 그런데 MAB 알고리즘에서도 supervised learning에서 나타나는 bias-variance tradeoff 문제가 발생할 수 있습니다. 평균이 낮은 regret도 중요하지만 variance가 낮은 regret도 중요하죠(기존 모델에서의 loss라고 생각하면 쉽습니다. Loss의 평균이 낮은 것도 중요하지만, variance가 낮아야 예측의 안정성을 보장합니다).	

## 3) Stationary

대부분의 모델에서의 가장 중요하고도 기본적인 가정은, data의 분포가 우리가 예측할 때와, 예측 모델을 학습할 때 일정한 분포라는 것입니다. 이를 **stationary** 라고 합니다. 하지만 MAB 환경을 살펴보면 이 조건을 만족하기가 힘듭니다. 가장 대표적인 예로, Supervised learning에서의 '개인지 고양이인지 맞추는 문제'와 MAB에서 '사용자에게 광고를 추천해주는 문제'를 생각해봅시다. 개인지 고양이인지는 시간이 지난다고 해서, 유행이 바뀐다고 해서 판단 기준이 바뀌지 않습니다. 그에 비해 사용자에게 광고를 추천해준다면, 어떤 트렌드가 유행인지, 계절은 어떤지, 고객들의 취향은 어떻게 바뀌는 지 등의 엄청나게 많은 변수에 따라 그 기준이 달라지게 됩니다. 따라서 이를 해결해주기 위해서, MAB에서는 크게 stationary bandit models와 non-stationary bandit models로 나집니다. 가장 간단한 방법은, 어떤 가치가 있을 때 이를 시간에 따라서 조금씩 decay해주는 방식입니다. 예를 들어 인기도라는 가치가 있다고 한다면, 그 인기도를 시간에 따라서 점차 줄어들게끔만 하는 것이죠.




## 4) Feedback Delay

다시 한번 이야기하지만 MAB는 온라인 피드백에서 강점을 가지고 있는 모델입니다. 온라인 모델이 기존 통계 모델과 다른 점은 데이터가 실시간으로 바뀌고, 환경이 실시간으로 바뀌고, 예측하고자 하는 값의 분포 또한 바뀔 수 있다는 점입니다. 따라서 이런 상황에서는 학습에 따른 feedback이 얼마나 빠르게 전달되는냐가 중요합니다. 아무리 좋은 모델이라도, feedback을 주는 와중에 환경이 아예 바뀌어 버린다면 좋은 feedback이 될 수가 없습니다.