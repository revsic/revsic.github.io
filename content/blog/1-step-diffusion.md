---
title: "[WIP] Essay: VAE is a 1-step Diffusion Model"
date: 2024-10-13T21:54:55+09:00
draft: false

# post thumb
image: "images/post/1-step-diffusion/vae.png"

# meta description
description: "Essay: VAE is a 1-step Diffusion Model"

# taxonomies
categories:
  - "Bayesian"
tags:
  - "Machine Learning"
  - "Deep Learning"
  - "Generative"
  - "Bayesian"
  - "Diffusion"
  - "Variational Lower Bounds"
  - "DDPM"
  - "VAE"
  - "Autoencoder"

# post type
type: "post"
---

아래 글은 비공식적인 개인의 사견임을 밝힌다.

- Essay on VAE and its relationship to diffusion model
- Keyword: VAE, Diffusion Model, VDM, VLB

**Introduction**

DDPM[[arXiv:2006.11239](https://arxiv.org/abs/2006.11239)] 이후 Diffusion Model은 그 합성 품질에 힘입어 빠른 속도로 발전해 왔다. 

DDPM은 Variational Lowerbounds(이하 VLB)를 통해 학습되고, 이는 2013년의 VAE[[arXiv:1312.6114](https://arxiv.org/abs/1312.6114)] 이후 꾸준히 활용되어 온 방법론이다.

그렇다면 DDPM은 어떻게 VAE 보다 더 High-fidelity의 이미지를 생성할 수 있었는가, 그에 대해 논의한다.

**Variational Lowerbounds**

VAE[[arXiv:1312.6114](https://arxiv.org/abs/1312.6114)]는 몇 가지 문제 상황을 가정한다.

어떤 데이터셋 $X = \\{x_i\\}^N_{i=1}$는 Random variable $x$에서 i.i.d.로 샘플링되었다. 우리는 이 데이터가 관측되지 않은 random variable $z$에 어떤 random process를 취해 생성하였다 가정할 것이다.

$z$는 prior distribution $p(z)$에서 샘플링되고, $x$는 조건부 분포 $p(x|z;\theta)$에 의해 생성된다. (그리고 각 분포는 z와 theta에 대해 미분가능하다 가정한다)

우리는 $p(z)$가 어떻게 생긴 분포인지 모르기 때문에, $p(x; \theta) = \int p(z)p(x|z; \theta)dz$의 marginalize가 불가능하다. (그렇기에 true posterior $p(z|x) = p(x|z)p(z)/(x)$ 역시 연산 불가능하다)

이에 대응하고자 approximate posterior $q(z|x; \phi)$를 도입하여 $\phi$와 $\theta$를 동시에 업데이트할 수 있는 objective function $\mathcal L$을 제안하였다.

$$\log p(x;\theta) = D_{KL}(q(z|x;\phi) || p(z|x;\theta)) + \mathcal L(x; \theta, \phi)$$
$$\mathcal L(x; \theta, \phi) = -D_{KL}(q(z|x; \phi)||p(z)) + \mathbb E_{q(z|x; \phi)}\left[\log p(x|z; \theta)\right]$$

$D_{KL}$은 0 이상 값을 가지므로 $\mathcal L(\theta, \phi; x)$는 log-likelihood의 하한이 되고, 이를 optimizing 하면 log-likelihood를 ascent 하는 것과 같은 효과를 볼 수 있다는 것이다.

DDPM 역시 Markov chain에 대한 variational lowerbound를 ascent 하는 방식으로 학습을 수행한다.

$x = x_0,\ z = x_T \sim \mathcal N(0, I)$의 T-step DDPM을 가정할 때, variance schedule $\beta_1, ...\beta_T$에 대해 forward process(noising) $q(x_t|x_{t-1})$와 reverse process(denoising) $p(x_{t-1}|x_t; \theta)$를 가정하자.

$$q(x_t|x_{t-1}) = \mathcal N(\sqrt{1 - \beta_t}x_{t-1} \beta_t I), \ p(x_{t-1}|x_t; \theta) = \mathcal N(\mu_\theta(x_t; t), \Sigma_\theta(x_t, t))$$

이때 VLB는 동일하게 적용된다.

$$\log p(x; \theta) \ge \mathbb E_{q(x_0|x)}[\log p(x|x_0)] + \mathcal L_{T}(x; \theta) - D_{KL}(q(x_T|x)||p(z))$$
$$\mathcal L_{T}(x; \theta) = -\sum^T_{i=1}\mathbb E_{q(x_i|x)} D_{KL}\left[q(x_{i-1}|x_i, x)||p(x_{i-1}|x_i; \theta)\right]$$

학습 목적 함수는 사실상 같다고 봐야 한다.

**Size of latent variables**

TBD

**VAE is a 1-step Diffusion Model**

TBD

**More step is better**

TBD

**Is diffusion model a 1,000-step VAE ?**

TBD

**References**

- VDM: Variational Diffusion Models, Kingma et al., 2021. [[arXiv:2107.00630](https://arxiv.org/abs/2107.00630)]
- NVAE: A Deep Hierarchical Variational Autoencoder, Vahdat & Kautz, 2020. [[arXiv:2007.03898](https://arxiv.org/abs/2007.03898)]
- DDPM: Denoising Diffusion Probabilistic Models, Ho et al., 2020. [[arXiv:2006.11239](https://arxiv.org/abs/2006.11239)]
- StyleGAN2: Analyzing and Improving the Image Quality of StyleGAN, Karras et al., 2019. [[arXiv:1912.04958](https://arxiv.org/abs/1912.04958)]
- VAE: Autoencoding Variational Bayes, Kingma & Welling, 2013. [[arXiv:1312.6114](https://arxiv.org/abs/1312.6114)]
