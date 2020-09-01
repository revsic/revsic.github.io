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

저자는 만약 이러한 규칙들을 직접 분석하여 수정할 수 있다면, 생성 모델 자체를 manipulating 하는 것이고, 이는 생성된 이미지를 각각 수정하는 것보다 효율적으로 수정된 이미지를 생성할 수 있다고 이야기 한다.

이를 위해서 우리는 생성 모델이 어떤 정보를 캡처하고 있고, 어떻게 unseen scenario에 대해 일반화 하고 있는지 알아야 한다.

현재 생성 모델들은 인간이 직접 라벨링 한 대규모의 데이터셋에 기반을 두고 있는데, 만약 manipulating 과정에서도 이러한 다량의 데이터와 학습이 추가로 필요하다면, 이는 손으로 생성된 이미지를 직접 수정하는 것과 큰 차이가 없을 것이다.

이에 우리는 단 몇 개의 샘플 데이터와 간단한 optimization을 통해 모델을 manipulation 할 수 있어야 하고, 이 모델은 우리가 원하는 rule을 캡처하여 unseen data에 대한 일반화를 할 수 있어야 한다.

저자는 이를 위해 sequential 하게 구성된 nonlinear convolutional generator를 associative memory라는 관점으로 해석하고, 전체 레이어가 아닌 단 하나의 레이어에 constrained optimization을 진행하여 우리가 원하는 rule을 추가할 수 있는 방법론을 제시한다. 

**Implementation**

- pytorch, official: [rewriting](https://github.com/davidbau/rewriting)

**References**

1. Progressive Growing of GANs for Improved Quality, Stability, and Variation, Tero Karras et al., 2017, [arXiv](https://arxiv.org/abs/1710.10196)