---
title: "Improving Deep Neural Networks - Hyperparameter Tuning"
excerpt: 이 글에서는 Deep Neural Network의 성능을 개선하는 방법에 대해 소개합니다.
date: 2018-05-23
layout: post
category:
- Case Study
tag:
- model tuning
---

## 들어가며

최근 머신러닝을 활용한 프로젝트들이나 kaggle competition들은 기존 statistical machine learning이나 boosting methods에서 벗어나 딥러닝을 활용한 프로젝트들이 주를 이루고 있습니다. 저또한 이제 막 딥러닝을 활용한 다양한 프로젝트를 시도해보고 있는데, 세상엔 좋은 모델들이 많지만 모델을 내가 활용하고자 하는 특정 데이터에 잘 튜닝하는 것이 많이 어려웠습니다. 이 글에서는 cousera의 [Improving Deep Neural Networks : Hyperparameter Tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network?authMode=login&errorCode=invalidCredential) 강의를 기반으로 어떻게 모델을 잘 최적화하는 지에 대한 방법들을 소개합니다. 추가적으로 자료를 찾아보면서 더 많은 내용을 담으려고 했습니다. 

> #### 주의
> 이 글에서는 특정 모델에 대한 이론적 내용이나 상황에 따라 어떤 모델을 선택해야 할 지에 대해서는 다루지 않습니다(상황, 과제에 따라 달라지기 때문에). 어떤 neural network모델을 정했다고 가정하고 이를 tuning할 수 있는 방법들을 정리한 것이라고 보면 됩니다.

#### 목차
- [Gradient Checking](#gradient-checking)
- [Initialization](#initialization)
- [Optimization](#optimization)
- [Fine Tuning Techniques](#fine-tuning-techniques)
- [Dropout and Activation Function](#dropout-and-activation-function)

---

## 1. Initialization
 Neural network를 학습시키려면 먼저 weights와 bias의 초기값들을 잘 선정해주어야 합니다. 초기값을 잘 정해놓기만 해도 더 효율적으로 학습할 수 있을 뿐만 아니라 실제로 더 좋은 성능을 보여준다고 합니다. 아직까지도 어떤 방법이 제일 좋은 방법인지는 증명되지 않았다고 합니다. 다만 실험적으로 몇몇 좋은 초기화 방법들은 존재합니다. Initialization 방법들에는 다음과 같은 방법들이 있습니다.
    - Zeros / Random initialization
    - Xavier initialization
    - He initialization
    - LSUV initialization (All you need is good init)

### (1) Zeros initialization
 가장 먼저 생각해볼 수 있는 초기화 방법들입니다. Weights를 0으로 초기화 할 경우 그냥 logistic regression이 되어버립니다. 유사한 문제로 symmetry problem이 있는데, 각 레이어의 각각의 뉴런이 같은 것을 배워서 똑같이 최적화되는 문제를 의미합니다. Bias를 0으로 초기화하는 것은 일반적인 방법 중 하나이므로 괜찮습니다. LSTM의 경우 bias를 1로 두는 방법도 있다고 합니다.

### (2) Random initialization
 Standard normal distribution에서 random한 값을 뽑아 그 값으로 weight를 정하는 것입니다. 그럴 경우 symmetry problem은 해결하지만 다른 2가지의 문제가 생길 수 있습니다. 바로 vanishing gradients problem과 exploding gradients problem입니다. 딥러닝을 최적화하는데에 있어서 가장 일반적으로 만나게 되는 문제들이기도 한데요, 각각의 문제들의 정의와 다른 관점에서의 해결 방법들은 다음을 참고하시기 바랍니다.
    - [Vanishing Gradient Problem](https://www.wikiwand.com/en/Vanishing_gradient_problem)
    - [Exploding Gradient Problem](https://machinelearningmastery.com/exploding-gradients-in-neural-networks/)
Random initialization의 또다른 문제는 바로 random성을 결정할 난수 값의 따라 성능이 크게 달라진다는 점입니다. 아래 그림처럼 같은 네트워크에 같은 활성화함수(ReLU) 같은 데이터라도 어떤 시드 값(random seed)으로 random initialization하느냐에 따라 성능과 수렴 시간이 달라집니다.
![](../assets/2018-05-23/1.png)

### (3) Xavier initialization
 Xavier initialization은 random initialization에서 발생하는 두가지 문제(vanishing & exploding gradient problem)을 어느정도 해결하였으며, 현재 가장 일반적으로 자주 쓰이는 초기화 방법입니다. 2010년에 나온 논문인 [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)에서 처음 소개되었으며, 저자인 Xavier glorot의 이름을 따서 Xavier initialization이라고 불립니다. 그 전에 성능 향상을 위한 다양한 초기화 방법들이 제안되었지만, 대부분 너무 계산량이 많고 구현하기 어려웠습니다. Xavier initialization은 다음과 같이 코드 몇줄이면 구현 가능합니다!

~~~python
# Xavier initialization
W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in)
~~~

 간단히 살펴보면, 입력값(`fan_in`)과 출력값(`fan_out`) 사이의 난수를 정한다음에 입력값의 제곱근으로 나눠준 값으로 초기화합니다.

### (4) He initialization
 He initialization은 Xavier와 거의 유사한데, `fan_in`을 넣는 대신 `fan_in/2`를 넣어줍니다.

~~~python
# He initialization
W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in/2)
~~~

### (5) LSUV initialization
 2015년에 나온 논문인 [All you need is good init](https://arxiv.org/pdf/1511.06422.pdf)에서 처음 소개된 방법으로, 다음의 2가지 순서를 통해 weight를 초기화합니다.
1. 각각의 convolution이나 inner-product layer를 orthonomal 한 행렬로 초기화합니다.
2. 첫 레이어부터 마지막 레이어까지 차례대로 돌면서 각각의 layer의 출력값의 분산이 1이 되도록 정규화해줍니다.

서로 다른 activation function(maxout, ReLU-family, tanh)에 대해 기존 초기화 방법들보다 더 좋은 성능을 보여주었다고 합니다. 성능 확인은 GoogLeNet, FitNets, Residual nets와 MNIST, CIFAR-10/100 데이터셋에서 최고 성능을 내는 네트워크들에게 이루어졌습니다. Pytorch code는 나와있는데([github](https://github.com/ducha-aiki/LSUV-pytorch)) 아직 tensorflow code는 없습니다. (왜일까요? 구현하기 어려워보이지는 않는데..)

> 찾아보니깐 누군가가 구현한 tensorflow코드는 있네요. 다만 따로 library형태는 아니라 성능 보장은 보장할 수 없습니다.
>
> https://github.com/PAN001/All-CNN/blob/8b3f20c3f62ef631c7d84482c48e9811da7e094b/strided_all_CNN_tf_LSUV.py

---
## 2. Before learning: sanity checks Tips/Tricks
일단 모델이 제대로 돌아갈 것인지를 판단해야 합니다. 다음 3가지 방법으로 이를 판단할 수 있습니다.
### (1) Look for correct loss at chance performance
쉽게 말해서 baseline이 무엇인지를 파악하는 것입니다. 예를 들어 결과값의 class가 10개인 문제에서는 맞을 확률이 찍어도 0.1이고, 이 때 활성화함수로 softmax layer를 쓰고 loss가 negative log probability라면 baseline은 $-ln(0.1)=2.302$가 됩니다. 초기 loss가 이보다도 낮다면 초기화에 문제가 있다고 볼 수 있습니다.
### (2) Increasing regularization should increase the loss
정규화를 더 세게 할수록 loss는 증가해야 합니다. 그렇지 않다면 애초에 tuning하기 전에 다른 문제들이 있을 수 있습니다.
### (3) Overfit a tiny subset of data
작은 데이터(약 20개)를 매우 overfitting 시킨다면 결국 loss는 0이 되어야 합니다. 이 때 정규화나 dropout은 사용하지 않고 해봐야 합니다.

## 3. Babysitting the learning process
모델을 train할 때 유심히 봐야할 몇가지 지표들이 있습니다. 여기서는 그 지표들에 대해 소개하고, 정상적으로 train된다면 어떻게 되어야 할 지에 대해 소개합니다.
### (1) Loss function

<img src="/images/2018-05-23-improving-deep-neural-networks/3.jpeg" alt="drawing" width="300"/> <img src="/images/2018-05-23-improving-deep-neural-networks/4.jpeg" width="300"/>

먼저 loss입니다. Train할 수록 떨어지는 것은 당연한데, 떨어지는 모양에 따라 **무엇이 문제인지** 알 수 있습니다. 위의 왼쪽 그림에서 빨간색 선처럼 loss가 줄어들어야 정상인데, 너무 갑자기 overfitting되어 올라가면 learning rate가 너무 큰 것이고, 너무 천천히 내려가면 learning rate가 작은 것입니다. learning rate를 잘못 선정하면 최종 loss 자체가 줄어들 수 있으니 잘 모니터하고 선정해야 합니다. 또한 보통 loss는 위의 오른쪽 그림처럼 진동(wiggle)하기 마련인데, 진동하는 정도는 바로 **batch size**와 관련이 있습니다. Batch size가 작을수록 진동하는 정도는 더 커집니다.

### (2) Train/Val accuracy
 두번째로 중요하게 봐야할 지표는 train / validation accuracy입니다. 아래 그림처럼 train accuracy는 validation accuracy에 비해 높은 것은 당연하지만, 지나치게 높으면 train set에 overfitting되고 있다는 증거이고, 지나치게 낮으면 underfitting되있다는 것입니다. 만약 파란색 선처럼 너무 overfitting되었다면 regularization을 증가하거나(L2 weight 증가, dropout rate 증가) 더 많은 데이터를 확보해야 합니다. 만약 초록색 선처럼 매우 유사하게 증가한다면 모델의 파라미터 수를 증가해줄 수 있습니다.
 
![accuracy](/images/2018-05-23-improving-deep-neural-networks/accuracies.jpeg)

### (3) Ratio of weights:updates
 마지막으로 gradient의 **변화량**입니다. 각각의 파라미터 셋에 대해 평가하는데, 만약 이 변화량이 $1e-3$ 근처에 있다면 정상이고, 그보다 낮다면 learning rate가 너무 낮다는 증거입니다. 

### (4) Activation/Gradient distributions per layer
 잘못된 초기화는 전체 학습과정을 상당히 느리게 할 수 있습니다. 이 문제는 쉽게 진단할 수 있는데요, 전체 layer에 대해서 activation/gradient histogram을 그려보면 됩니다. 그 값이 -1에서 1 사이에 분포되어 있으면 정상이고 이상한 값이나 다 0으로 되어 있으면 문제가 있는 것입니다.

### (5) First layer visualizations
특히 **이미지 처리** 관련 분야에서는 첫번째 layer가 어떤 결과를 내는지 시각화해서 보면 대략적으로 잘 작동되고 있는지 볼 수 있습니다.

<img src="/images/2018-05-23-improving-deep-neural-networks/weights.jpeg" width="250"/> <img src="/images/2018-05-23-improving-deep-neural-networks/cnnweights.jpg" width="280"/>

왼쪽 그림을 보면 딱봐도 noise가 많아보이므로, learning rate가 잘못 선정되었거나 regularization이 너무 적을 수 있습니다. 반면 오른쪽 그림은 다양한 feature를 잡아내는 것으로 볼 수 있어 train이 잘 되고 있다고 보면 됩니다.

## 4. Parameter updates
계산된 loss와 gradient를 기반으로 optimization method들을 어떻게 시도하고 선정할 지에 대한 부분입니다. 다만 이 부분은 지금도 연구가 활발하게 진행되고 있는 분야이기 때문에, 어떤 방법이 정답이라기 보다는 제일 일반적이고 기본적인 관점과 접근을 소개합니다. 

### (1) SGD with bells and whistles
- Vanilla update : $$x += lr * dx$$
- Momentum update : $$v = mu * v -lr * dx, x += v$$
- Nesterov momentum : 

### (2) Annealing the learning rate
learning rate는 시간이 갈수록 줄여야 합니다. 쉽게 생각해서 learning rate가 크면 parameter vector가 극심하게 왔다갔다하기 때문에 깊은 자리에 들어가기 힘듭니다. 따라서 점차 learning rate를 줄여나가야 하는데, 그 일반적인 방법에는 다음이 있습니다.
- Step decay : 몇번의 epoch마다 일정한 값을 줄입니다.
- Exponential decay : 몇번의 epoch마다 일정한 비율로 줄입니다.
- 1/t decay : $$\alpha = \alpha_0 / (1+kt)$$
https://www.jeremyjordan.me/nn-learning-rate/

### (3) Per-parameter adaptive learning rates
지금까지 논의된 접근법들은 모든 파라미터에 똑같은 학습 속도를 적용하였습니다. 이를 해결하고 데이터에 맞추어 자동으로 학습속도를 정하는 방법을 찾고자 많은 사람들이 노력하였습니다. 
- Adagrad : 데이터에 맞춘 학습속도 조정방법

~~~python3
cache += dx**2
x += -lr * dx / (np.sqrt(cache)+eps)
~~~

- RMSprop
- Adam

## 5. Hyperparameter optimization
이제 학습에서 가장 빈번하게 조절해야할 hyperparameter를 어떻게 수정해나가면서 찾을 것인지를 알아보겠습니다. Neural network에서 가장 중요한 hyperparameter는 다음과 같습니다. 
- initial learning rate
- learning rate decay schedule
- regularization strength(L2 penalty, dropout strength)

### Initial learning rate
가장 중요한 점은 **log scale**로 늘리고 줄여야 한다는 점입니다. 보통 `1e-3`이나 `3e-4`를 결정하여 쓰지만, 이는 어떤 optimizer가 제일 좋다라는 말만큼이나 말도 안되는 얘기고, 이상적인 값은 없습니다.

---

## Reference
- [Learning Neural Networks](http://cs231n.github.io/neural-networks-3/)
- [An overview gradient descent algorithms](http://ruder.io/optimizing-gradient-descent/index.html)
- [Optimization for Deep Learning Highlights in 2017](http://ruder.io/deep-learning-optimization-2017/)
