---
title: "Deep Learning Glossary"
excerpt: WildML의 [Deep Learning Glossary](http://www.wildml.com/deep-learning-glossary/) 정리해보았습니다.
date: 2018-03-13
layout: post
category:
- blog
tag:
- glossary
- wildml
---

- Activation function
    
    딥러닝의 가장 큰 장점인 **복잡한 decision boundary**를 적용하기위해, 비선형적인 activation function을 layer에 추가합니다. 보통 `sigmoid`, `tanh`, `ReLU`와 이들의 변형을 사용합니다.

- Adadelta

    Gradient descent을 기반으로 만든 알고리즘으로 파라미터마다 learning rate를 최적화합니다. Adagrad를 개선하기 위해 만들어졌으며, hyperparameter에 더 민감합니다. Adadelta는 rmsprop과 유사하며 vanilla SGD 대신 사용되기도 합니다.
    
- Adagrad
    
    Adagrad는 adaptive learning rate algorithm 방법들 중 하나로 시간에 따라 바뀌는 griadients의 squared 값을 추적하고, 동시에 learning rate를 최적화시킵니다. Vanilla SGD 대신 쓰이기도 하며, 특히 sparse한 데이터에 좋습니다. 
    
- Adam
    
    Adam도 optimizing algorithm 중 하나로, rmsprop과 유사하지만 update를 첫번째와 두번째 순간의 gradient의 running average로 추정합니다. 또 bias correction term도 포함되어 있습니다.
    
- Affine layer

    **Fully-connected layer**와 같은 말입니다. 그 전 레이어의 모든 뉴런이 현재 레이어의 모든 뉴런과 연결되어 있다는 뜻입니다. 보통 output의 final prediction전에 추가됩니다.
    
- Attention mechanism

    **Human visual attention**에서 영감을 받았으며, 어떤 이미지에서 특정 부분에 더 집중하게 됩니다. Language processing과 image recognition 구조에 모두 적용됩니다. 결국 예측할 때, 어느 부분에 집중하여 예측할 지를 결정하는 것입니다.
    
- Alexnet

    CNN architecture의 일종으로 ILSVRC 2012에서 우승하였으먀 5개의 convolution layer와 max-pooling을 추가하였고 3개의 affine layer가 추가되어 finally 1000 way softmax가 추가되어있는 형태입니다.
    
- Autoencoder

    Autoencoder는 목표가 input을 그대로 예측하는 것입니다. **Bottleneck** 구조를 띄고 있으며, 그를 위해서 일단 저차원으로 represent하고 이를 다시 복원합니다. 따라서 PCA와 같은 다른 차원축소 기법들과 유사하지만, 비선형적인 환경으로 인해 더 복잡한 mapping이 가능합니다. 변형 기법으로 Denoising Autoencoders, Variational Autoencoders, Sequence Autoencoders 등이 있습니다. 
    
- Average-pooling

    Pooling 기법 중 하나로 CNN for image recognition에 쓰입니다. 정해진 크기의 window(filter)를 이동시키면서 그 평균을 구합니다. 뭐, input을 저차원으로 압축시킨다고 보면 됩니다.
    
- Backpropagation through time

    역적판 기법 중 하나로 RNN에 쓰입니다. 표준의 역전파 방식을 RNN에 적용시켰다고 보면 되고, layer가 시간을 의미하기 때문에, 시간을 관통하는 역전파라고 불리는 것입니다. 몇백개의 input이라고 한다면, BPTT가 computational cost를 줄이기 위해 사용됩니다.
    
- Batch normalization

    작은 batch마다의 layer input들을 normalize하는 방법입니다. 학습을 더 빠르게 하고, learning rate를 더 빠르게 최적화하며, regularizer 역할도 해줍니다. CNN에는 좋지만 RNN에는 안좋다고 합니다.
    
- Bidirectional RNN

    두 개의 RNN을 반대 방향을 연결한 네트워크 구조입니다. Foward RNN은 input sequence를 순방향으로 읽고, Backward RNN은 역방향으로 읽습니다. NLP에 자주 쓰이고, 그 단어 전후의 문맥을 파악하고자 할때 쓰입니다.
    
- Categorical cross-entropy loss

    **Negative log likelihood**와 같은 말입니다. 분류 문제에서 유명하고, 두 개의 확률분포에서의 유사성을 측정합니다. $L=-\sum(y*\log(y_{prediction}))$
    
- Channel

    input data의 channel은 여러 개일 수 있습니다. 이건 뭐..
    
- Deep belief network

    DBN은 probabilistic graphic model 중 하나로 data의 hierachical representation을 비지도학습으로 배웁니다. 
    
- Deep dream
    
    CNN에서 포착된 knowledge를 시각화하는 기법으로, 기존의 이미지를 변형하거나 새로운 그림을 만들면서 이를 꿈같은 표현으로 생성합니다..??
    
- Dropout

    regularization technique의 일종으로 overfitting을 방지합니다. 임의로 가지치기를 한다고 생각하시면 됩니다. 
- Embedding

    input을 vector로 표현하는 방법입니다. word2vec같이 explicit하게 배우거나, sentiment analysis 같이 지도 학습을 통해 배워지기도 합니다. pre-trained embedding에서 시작되어 그 과제에 맞게 fine-tuning되기도 합니다.
    
- Exploding gradient problem
    
    Vanishing Gradient Problem의 반대의 의미입니다. Gradient clipping을 통해 이를 조절합니다.
    
- Fine-tuning

    다른 과제 또는 환경에서 학습된 네트워크를 원하는 과제에 맞게 다시 최적화하는 방법입니다. 
    
- Gradient clipping

    exploding gradients를 조절하기 위해 사용하는 방법으로, 특히 RNN에서 자주 사용됩니다. 대표적인 방법으로는 L2 norm이 threshold를 넘을 때 normalize하는 방법입니다.
    $$
    Gradients_{new} = \frac{Gradients_{old} * threshold}{l2}
    $$
- GloVe

    embedding 기법 중 하나로, co-occurence matrix의 통계량을 이용한다는 특징이 있습니다.
    
- GoogleNet

    ILSVRC 2014에서 우승한 CNN architecture입니다. Inception module을 사용하여 네트워크의 computing resource를 향상시켰습니다.
    
- GRU

    Gated Recurrent Unit의 약자입니다. LSTM의 간단 버전이고 파라미터 수가 더 적습니다. LSTM과 마찬가지로 RNN에서 gating mechanism을 써서 long-range dependency를 조절하고 vanishing gradient problem을 해결합니다. GRU도 reset과 update gate가 있어서 old memory중 어떤 부분을 보존하거나 update할 지를 결정합니다.
    
- Highway Layer

    gating mechanism을 전체 layer에 적용시킨 모델입니다. 아주 깊은 모델 구조에서 input의 어떤 부분을 그대로 넘기고, 어떤 부분을 변형해서 넘길지를 결정합니다. 
    $$
    T\times h(x)+(1-T)\times h(x)
    $$
- ICML

    International Conference for Machine Learning
    
- ILSVRC
    
    ImageNet Large Scale Visual Recognition Challenge

- Inception module

    CNN architecure에서 더 효율적인 계산과 더 깊은 네트워크를 위해 쓰이는 방식으로 차원 축소와 stacked 1x1 convolutions를 사용합니다.
    
- LSTM

    vanishing gradient problem을 해결하기 위해서 발명되었으며 memory gating mechanism을 사용합니다.
    
- Max-pooling

    CNN에서 사용되는 pooling 기법들 중 하나입니다. 이름만 봐도 뭔지 알겠죠?

- Momentum

    Gradient descent algorithm의 확장으로, parameter 갱신을 가속화하거나 감속합니다. Gradient descent를 업데이트할 때, momentum term을 넣으면 더 잘 수렴합니다.
    
- Neural machine translation

    언어간 번역을 의미합니다. end-to-end bilingual corpora를 통해 학습되기도 하며, 기본적으로는 encoder와 decoder rnn을 사용합니다.
    
- Neural turing machine

    simple algorithm을 neural network로 구현하는 겁니다.
    
- Noise-contrastive estimation

    large output vocabulary를 가진 분류 모델을 학습시킬 때 쓰일 수 있는 sampling loss입니다. 모든 가능한 경우의 수에 대해 softmax를 쓰기는 어렵습니다. NCE를 사용하면, 문제를 binary classificaion problem으로 제한할 수 있습니다. 
    
- Restricted boltzmann machine

    probabilistic한 graphic model의 일종으로 stochastic artifical neural network이다. 비지도 학습으로 데이터를 vector로 바꿉니다. Contrasive divergence를 사용하여 더 효율적으로 학습합니다.
    
- Recursive nerual network

    RNN을 트리 구조에 적용한 모델입니다. 이미 잘 train된 rnn에 사용되곤 합니다. parsing tree in NLP
    
- ResNet

    ILSVRC 2015에 우승한 CNN architecture입니다. layer를 건너뛰어서 연결하는 residual mapping을 이용하였습니다.
    
- RMSProp

    gradient-based optimization algorithm 기법 중 하나입니다. adagrad와 유사하지만, decay term을 추가함으로서 adagrad의 learning rate의 급격한 감소를 방지합니다.
    
- Seq2Seq

    Sequence-to-sequence model
    
- SGD

    Stochastic Gradient Descent입니다. 보통 minibatch에서 사용하곤 합니다. 
    
- Vanishing gradient problem

    너무 깊은 neural network에서 발생하는 문제 중 하나입니다. 역전파 과정에서 작은 gradient가 곱해져서 없어지는 경향을 말합니다. 이를 해결하기위해 ReLU를 쓰거나, LSTM같은 architecture를 쓰거나 합니다. 반대의 경우엔 exploding gradient problem이 발생합니다.
