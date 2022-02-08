---
title: RNN initHidden
layout: post
Created: January 25, 2022 1:16 PM
tags:
    - Pytorch
comments: true
use_math: true
sitemap :
  changefreq : daily
  priority : 1.0
---

RNN의 initial hidden state에 대해 default approach로 zero state를 사용한다. 이는 initial state에 영향을 많이 받는 ouput의 비율이 작은 seq-to-seq LM에서는 잘 작동한다. 하지만, (1) initial state를 model parameter로 학습하거나 (2) noisy initial state를 사용하거나 (3) 둘다 활용할 수 있다.

## (1)**Training the initial state**

“enough sequences or state resets in the training data”인 경우에 적절하다. 모든 step에 대해 학습하는 것이 아니라 n-step을 지정하여 initial state를 학습한다. 경험적으로 어느 것이 더 좋은지 판별 할 수 밖에 없다.

## (2)**Using a noisy initial state**

zero-valued initial state를 사용하면 overfitting이 생길 수 있다. 일반적으로 seq-to-seq model의 초기 step에서 loss는 추후 step의 loss보다 더 클 것이다. 왜냐하면 history가 적기 때문에.

> if all state resets are associated with a zero-state, the model can (and will) learn how to compensate for precisely this.
>

모든 state를 zero-state와 associated하면, model은 정확하게 zero-state에 대해 보상하는 방법을 학습할 수 있을 것이다...?

## **Empirical results**

![Untitled](RNN%20initHidden%20c44a05ced6154f8eb8cd2f78ae73be00/Untitled.png)

![Untitled](RNN%20initHidden%20c44a05ced6154f8eb8cd2f78ae73be00/Untitled%201.png)

→ 해당 링크에서 실험한 결과, 모든 none-zero state initialization이 학습속도도 빠르고 일반적으로 향상되는 모습을 볼 수 있다. 또, initial state를 학습한 것이 0을 평균으로 하는 noisy initial state보다 효과적이다. 마지막으로 (3)의 경우는 미미한 benefit만 제공한다.

- All non-zero state intializations sped up training and improved generalization.
- Training the initial state as a variable was more effective than using a noisy zero-mean initial state.
- Adding noise to a variable initial state provided only marginal benefit.

---

[How/What to initialize the hidden states in RNN sequence-to-sequence models?](https://datascience.stackexchange.com/questions/27225/how-what-to-initialize-the-hidden-states-in-rnn-sequence-to-sequence-models)

[Non-Zero Initial States for Recurrent Neural Networks - R2RT](https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html)

- Forecasting with Recurrent Neural Networks:12 Tricks

[https://www.scs-europe.net/conf/ecms2015/invited/Contribution_Zimmermann_Grothmann_Tietz.pdf](https://www.scs-europe.net/conf/ecms2015/invited/Contribution_Zimmermann_Grothmann_Tietz.pdf)
