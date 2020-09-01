---
title: "Rewriting a Deep Generative Model"
date: 2020-09-01T14:09:08+09:00
draft: true

# post thumb
image: "images/post/rewriting/1.jpg"

# meta description
description: "Rewriting a Deep Generative Model, David Bau et al., 2020."

# taxonomies
categories:
  - "Generative"
tags:
  - "Machine Learning"
  - "Deep Learning"
  - "Generative"
  - "Adversarial Learning"
  - "Model editing"
  - "Rewriting"

# post type
type: "post"
---

- David Bau et al., 2020, [arXiv](https://arxiv.org/abs/2007.15646)
- Keyword: Generative, Adversarial learning
- Problem: How to manipulate specific rules encoded by a deep generative model.
- Solution: Projected gradient descent for adding rules to convolution of associative memory.
- Benefits: Enable users to synthesize edited new images by manipulating model only once.
- Contribution: Providing a new perspective of associative memory, rule manipulating method of projected gradient descent.
- Weakness or Future work: -

**Generative model**

생성 모델은 데이터의 분포를 학습하면서 여러 가지 규칙이나 관계를 만들어 나간다. 간단한 예로 ProgressiveGAN[1]이 만든 주방 이미지에서는 창문에서 오는 빛을 테이블에 반사시키는 경향이 있다. 

{{< figure src="/images/post/rewriting/2.jpg" width="100%" caption="Fig. 6: Inverting a single semantic rule within a model" >}}

저자는 만약 이러한 규칙들을 직접 분석하여 수정할 수 있다면, 생성 모델 자체를 manipulating 하는 것이고, 이는 생성된 이미지를 각각 수정하는 것보다 효율적으로 수정된 이미지를 생성할 수 있다고 이야기 한다.

이를 위해서 우리는 생성 모델이 어떤 정보를 캡처하고 있고, 어떻게 unseen scenario에 대해 일반화 하고 있는지 알아야 한다.

현재 생성 모델들은 인간이 직접 라벨링 한 대규모의 데이터셋에 기반을 두고 있는데, 만약 manipulating 과정에서도 이러한 다량의 데이터와 학습이 추가로 필요하다면, 이는 손으로 생성된 이미지를 직접 수정하는 것과 큰 차이가 없을 것이다.

이에 우리는 단 몇 개의 샘플 데이터와 간단한 optimization을 통해 모델을 manipulation 할 수 있어야 하고, 이 모델은 우리가 원하는 rule을 캡처하여 unseen data에 대한 일반화를 할 수 있어야 한다.

저자는 이를 위해 sequential 하게 구성된 nonlinear convolutional generator를 associative memory라는 관점으로 해석하고, 전체 레이어가 아닌 단 하나의 레이어에 constrained optimization을 진행하여 기존의 semantic rule을 보존하면서, 우리가 원하는 rule을 추가할 수 있는 방법론을 제시한다. 

**Preview**

pretrain된 generator $G(\cdot; \theta_0)$가 주어질 때, 모델은 각각의 latent $z_i$에 대해 $x_i = G(z_i; \theta_0)$의 output을 만들어 낸다. 만약 우리가 copy&paste 방식으로 변화를 준 output $x_{*i}$을 통해 새로운 rule을 표현한다면, rule의 표현 중 가장 직관적인 방법일 것이다.

{{< figure src="/images/post/rewriting/3.jpg" width="100%" caption="Fig. 3: The Copy-Paste-Context interface for rewriting a model." >}}

이때 하고자 하는 것은 새로운 rule을 따르는 $\theta_1$을 만드는 것이고, 이는 $x_{*i} \approx G(z_i; \theta_1)$을 만족할 것이다.

$\theta_1 = \arg\min_\theta \mathcal L_{\mathrm{smooth}}(\theta) + \lambda \mathcal L_\mathrm{constraint}(\theta)$

$\mathcal L_\mathrm{smooth}(\theta) \overset{\Delta}{=} \mathbb E_z[\mathcal l(G(z; \theta_0), G(z; \theta))]$

$\mathcal L_\mathrm{constraint}(\theta) \overset{\Delta}{=} \sum_i \mathcal l(x_{*i}, G(z_i; \theta))$

고전적인 해결책은 generator의 전체 parameter set $\theta_0$를 두 가지 constraint에 맞게 gradient 기반의 optimization을 진행하는 것이다. 이때 $\mathcal l(\cdot)$은 perceptual distance를 의미한다. 

하지만 이 경우 몇 개 되지 않는 sample에 overfit될 가능성이 농후하며 다른 데이터에 대해 일반화되지 않을 수 있다.

이에 저자는 두 가지 방법론을 제안한다. 하나는 전체 parameter set이 아닌 특정 한 layer의 weight만을 update하는 것이고, 하나는 optimization을 특정 constraint 내에서 진행하는 것이다.

특정 layer L과 L-1 layer까지의 feature map k를 가정할 때 L의 output은 $v = f(k; W_0)$가 된다. 원본 이미지의 latent $z_{i}$가 feature $k_{*i}$를 만들 때 $v_i = f(k_{*i}; W_0)$를 가정하고, 직접 수정한 output에 대응하는 feature map $v_{*i}$를 구할 수 있으면 objective는 다음과 같다.

$W_1 = \arg\min_W \mathcal L_{\mathrm{smooth}}(W) + \lambda \mathcal L_\mathrm{constraint}(W)$

$\mathcal L_\mathrm{smooth}(W) \overset{\Delta}{=} \mathbb E_z[|| f(k; W_0) - f(k; W)||^2]$

$\mathcal L_\mathrm{constraint}(W) \overset{\Delta}{=} \sum_i ||v_{*i} - f(k_{*i}; W)||^2$

perceptual distance는 higher semantic을 표현하는 feature map 사이의 l2-distance를 상정한다. 이때 W만으로도 parameter의 양이 충분히 많을 수 있기에, overfit을 제한하면서 더 나은 일반화를 위해 학습 방향을 고정할 필요가 있었고, 특정 direction으로만 optimization 되도록 constraint를 추가한 gradient descent를 진행하게 된다.

**Associative Memory**



**Implementation**

- pytorch, official: [rewriting](https://github.com/davidbau/rewriting)

**References**

1. Progressive Growing of GANs for Improved Quality, Stability, and Variation, Tero Karras et al., 2017, [arXiv:1710.10196](https://arxiv.org/abs/1710.10196)