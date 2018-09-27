---
title: "A Persona-Based Neural Conversation Model"
excerpt: 대화 생성 모델에서의 응답자의 정보를 동일하게 유지하기 위한 방법에 관한 논문을 읽고 정리했습니다.
date: 2018-03-08
layout: post
category:
- [Natural Language Processing, Chatbots]
- Paper Review
tag:
- persona
- chatbots
- Seq2Seq
---

# A Persona-Based Neural Conversation Model

대화 생성 모델에서의 응답자의 정보를 동일하게 유지하기 위한 방법을 소개합니다. 이 방법은 Seq2Seq 모델을 기반으로 만들어졌습니다. Speaker Model과 Speaker-Addressee Model 이 있는데요, Speaker Model은 인물의 개성을 추출하여 이를 벡터화 하는 모델이구요, Speaker-Addressee Model은 두명의 발화자의 상호작용을 통해 추가적인 정보를 추출해냅니다. 

### 1. Introduction
보통 data-driven system은 likelihood가 가장 높은 순서로 response를 정합니다. 그러나 그냥 *비슷한 결과* 만을 도출하기 때문에 일관성이 부족합니다. 집의 주소를 약간의 화법만 바꿔서 물어봐도 집 주소가 바뀝니다. 이를 해결하기 위해서 PERSONA를 정의하고 이를 추출하기 위해 노력합니다. PERSONA는 내재된 사실, 내재된 유저의 정보, 언어 형식, 상호작용 스타일 등이 될 수 있습니다. PERSONA는 또한 가변적이어야 합니다. 왜냐하면 상호작용이나 대화상대가 바뀜에 따라 persona도 바뀌어야 하기 때문입니다.

이를 두 개의 모델로 해결했는데요. 먼저 Speaker model은 speaker를 벡터로 표현한 다음 이를 SEQ2SEQ Target에 통합합니다. 비슷한 형태로, Speaker-Addressee model은 두 대화자의 interaction pattern을 Seq2Seq model에 결합합니다.
> 실제로 어떻게 넣었는지는 전혀 감이 안오지만, 일단 느낌만 알고 pass

### 2. Related Works
- SMT 모델 소개
- Seq2Seq 모델 소개
- Standard dialog model 소개

### 3. Seq2Seq Models : Pass

### 4. Personalized Response Generation
1. Notation
    - $M$ : input word sequence {$m_1, m_2, ..., m_I$}
    - $R$ : word sequence in response to $M$ {$r_1,r_2,...,r_J,EOS$}
    - $J$ : length of the response
    - $r_t$ : $K$차원의 word embedding $e_t$에서 나온 단어 토큰
    - $V$ : vocabulary size
2. Speaker Model

    Speaker의 특수정보(dialect, register, age, gender, personal information)를 벡터로 변환하는 모델입니다. 이러한 정보들은 인위적으로 조절되지 않았습니다. 대신, 유저들을 특정 정보에 따라 clustering해줍니다. 
3. Speaker-Addressee Model

    발화 형식이나 대화내용 등의 개성은 발화자 뿐만 아니라 "누구와 대화하느냐"에 따라서도 달라지게 됩니다. 이 모델은 발화자 i가 발화자 j와 대화할 때 어떻게 대화할지를 예측합니다. 이는 엄청 큰 크기의 매트릭스를 만들어야 한다는 단점이 있지만, 유저의 정보를 임베딩하는 과정에서 비록 유저 $i$와 유저 $j$가 만난 적이 없더라도, $V_{i'j']$를 계산하여 예측합니다. 여기서 $i'$, $j'$은 가장 비슷한 유저들을 의미합니다.
4. Decoding and Reranking
    Decoding을 위해서, beam size $B$=200인 decoder로 만들어낸 N-best list를 활용합니다. 그 다음에 이전에 소개했던 MMI를 활용하여 최고의 답변을 찾아냅니다.
    
---
