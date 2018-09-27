---
title: "Deep Learning for Chatbots"
excerpt: 이 글에서는 대화가능한 챗봇을 구현하기 위한 딥러닝 기법들을 소개합니다. 지금 어느 상태까지 발전되어 있고, 무엇이 가능하고 어느 것이 한동안은 불가능할지 살펴보는 것부터 시작합니다. WildML의 글을 참고하여 정리하였습니다.
date: 2018-03-06
layout: post
category:
- [Natural Language Processing, Chatbots]
tag:
- wildml
---

# Deep Learning for Chatbots, Part 1 - Introduction

### A taxonomy of models

#### Retrieval-based vs Generative Models
**Retrieval-based models**는 heuristic한 방법으로 미리 정해진 답변들을 mapping해줍니다. 그 과정에서 rule-based 모델이나 machine learning classifiers의 앙상블 모델을 사용하기도 합니다. 이 방법은 새로운 텍스트를 만들지 않고, 단지 이미 정해진 set에서 하나 고르게 됩니다.

**Generative Models**은 미리 정해진 답변에 의존하지 않습니다. 이들은 아예 새로운 답변들을 만들어 냅니다. 기본적으로 Machine translation techniques에 기반하여 만들어지지만, 하나의 언어를 다른 언어로 번역하는 것이 아니라, 하나의 input을 원하는 output으로 translate합니다.

두가지 방법 모두 장점과 단점들이 있습니다. Retrieval-based models는 데이터셋이 정해져 있기 때문에, 문법적인 오류가 나타나지 않습니다. 하지만 예상치못한 케이스나 미리 정해져 있는 질문과 조금만 달라도 답을 내는 것이 불가능하죠. 같은 이유로 상황에 대한 정보(이름)같은 것들을 전혀 이해하지 못합니다. 반면에 Generative models는 더 똑똑합니다. 잘만 하면 마치 사람이 답변하는 듯한 답변을 만들어낼 수 있습니다. 그러나 train하기 힘들고, 문법적인 오류들을 내기 쉬우며, 엄청나게 큰 데이터셋을 필요로 합니다.

딥러닝 기법들은 두가지 상황 모두에서 사용될 수 있지만, 여기서는 generative한 방향으로 소개합니다. Seq2Seq같은 모델은 텍스트를 생성하는 데에 나름 최적화되어 있지만, 제대로된 결과를 생성해내기까지는 시간이 좀더 걸릴 듯 합니다. 지금 기업들에서는 보통 retrieval-based models를 사용하고 있습니다.

#### Long vs Short Conversations
긴 대화일수록 이를 자동화하기는 더 어려워집니다.
- 쉬움 : Short-Text Conversations
- 어려움 : Long Conversations

### Open Domain vs Closed Domain
Open domain이 더 힘듭니다. 잘 정의된 목표도 없고, 대화가 어느 방향으로 흐를지도 잘 모릅니다. 무한한 가능성의 주제와, 그 주제가 주어지면 해당 주제에 대한 어느정도의 지식 또한 있어야 제대로된 답을 할 수 있습니다.

Closed domain은 가능한 input이 정해져 있고, output도 제한적입니다. 기술적 고객 지원이나 구매 관련 문의 등이 좋은 예죠.

---
### Common Challenges
다음은 대화가능한 챗봇을 만들 때 겪게 되는 명확한 vs 애매한 과제들입니다.

#### Incorporating Context - 맥락 다루기?
센스 있는 답변을 내기 위해서는 linguistic context와 physical context를 모두 다룰 수 있어야 합니다. 여기서 linguistic context는 긴 대화에서 무엇을 말했는지, 어떤 정보가 교환되었는지 등을 의미합니다. 가장 일반적인 방법은 대화를 벡터로 임베딩하는 것입니다. 

#### Coherent Personality 성향의 일관성
문법적으로 다르지만 같은 의미의 질문에 대해 동일한 답변을 해야한다.
== 문법적으로 비슷하지만 완전 다른 의미의 질문에 대해 다른 답변을 해야한다.

-> 이번에 고치려고 하는 문제와 많이 비슷합니다.

#### Evaluation of Models
대화가능한 챗봇을 평가하는 가장 이상적인 방법은 그들이 해결해야한 문제를 얼마나 잘 해결했는지 입니다. 하지만 이를 평가해서 라벨을 얻기란 매우 비쌉니다(인건비). BLEU같은 일반적인 행렬구조가 있기도 하지만 센스있는 답변을 평가하기엔 부족합니다. 사실, 그 어떤 매트릭스 구조도 제대로 평가하지 못한다고 증명한 사례가 있습니다.

#### Intention and Diversity
Generative model의 가장 평범한 문제는 '몰라요', '글쎄요'같은 일반적인 답변을 내기 쉽다는 것입니다. 어떤 연구는 이 문제를 다양한 목적함수를 합침으로서 좀더 다채롭게 하려고 시도했습니다. 그러나 사람은 보통 입력에 매우 한정된 답변을 하고 그를 위해서 특정한 의도를 가집니다. Generative model은 사람처럼 특정 의도를 가지지 못하기 때문에 이런 다양성을 가지기 힘듭니다.

---
### 그래서 실제로 어떻게 작동하나요?
- Retrieval-based open domain system : 불가능
- Generative open-domain system $\approx$ Artificial General Intelligence
-> 둘다 매우 힘듬

따라서 제한된 상황에서의 문제 해결들만 남게 되었습니다. 문장이 길어질수록, 문맥적 의미가 중요해질수록 더 문제는 어려워집니다.

앤드류 응님에 따르면,
> 딥러닝의 진가는 많은 양의 데이터를 얻을 수 있는 제한된 영역에서 나타납니다. 딥러닝이 절대 못하는 것은 "의미있는 대화 나누기"입니다. 의미 있는 대화를 만들기 위해 cherry-picking할 수는 있지만, 결국 다시 혼자 해보면 망합니다.

많은 기업들은 그들의 대화를 저장하고 외주를 맡기면 그들의 대답을 "자동화"할 수 있을 거라 기대합니다. 하지만 이는 그 대답이 충분히 "좁은 영역"일 때만 가능합니다. 하지만 그렇더라도 사람 노동자를 지원하기에는 충분합니다.

생산 시스템에서의 문법오류는 매우 큰 손실을 끼칠 수 있습니다. 그 것이 아직까지도 retrieved-based model들이 주류를 이루는 이유 중 하나입니다. 만약 기업들이 엄청나게 큰 데이터를 갖게 된다면 generative model도 구현가능해지겠지만, 심각한 오류가 나지 않도록 하는 다른 기술지원들이 병행되어야 합니다.

---

