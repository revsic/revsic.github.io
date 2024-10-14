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

VAE는 몇 가지 문제 상황을 가정한다.

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

VAE와 Diffusion Model의 차이로 떠오르는 것은 Bottleneck Architecture이다.

VAE는 latent variable의 dimension이 데이터보다 대개 작다. Diffusion은 markov chain 내의 state를 모두 latent variable로 바라보고, 각각의 latent variable은 데이터의 dimension과 크기가 같다.

작은 latent variable은 초기 GAN[[arXiv:1406.2661](https://arxiv.org/abs/1406.2661)] 기반의 모델에서도 공통으로 나타나는 특징이다.

이후 VAE와 GAN 모두, 데이터 차원과 같은 크기의 잠재 변수를 도입하여 성능 향상을 본 모델이 나온다. StyleGAN[[arXiv:1812.04948](https://arxiv.org/abs/1812.04948)]은 이미지의 stochastic detail을 생성하기 위해 $\mathbb R^{\mathrm{H\times W\times 1}}$의 single-channel noise를 더하였고, NVAE[[arXiv:2007.03898](https://arxiv.org/abs/2007.03898)]는 U-Net-like architecture를 도입하면서 residual signal을 latent variable로 모델링한다.

{{< figure src="/images/post/1-step-diffusion/1.png" width="100%" caption="Left: Figure 1, Karras et al.(StyleGAN), 2018 / Right: Figure 2, Vahdat & Kautz(NVAE), 2020." >}}

다만 둘 모두 이론적 근거를 제시하기보단 Ablation study를 통해 정량적, 정성적 개선 정도를 보인다.

이미지의 대략적인 형상과 배치 등 lower frequency의 정보는 작은 잠재 변수 공간에서 capture 할 수 있지만, Higher frequency의 정보를 capture 하기 위해서는 spatial information에 correlate 된 latent variable이 있어야 하지 않을까 싶은 정도이다.

**VAE is a 1-step Diffusion Model**

VAE가 이미지의 크기와 같은 잠재 변수를 취급하고, $z \mapsto x$의 매핑을 U-Net을 통해 모델링한다 가정하자.

동일하게 VLB를 통해 학습되고, 잠재 변수의 크기도 이미지의 차원과 같으며, U-Net을 백본으로 사용한다. 

이제 VAE는 T=1인 Single-step diffusion model이라 볼 수 있다. T=1 이므로 time embedding은 배제할 수 있고, 완전히 동일한 백본을 가정해도 무방하다.

Generation process도 동일하다. 1-step DDPM은 $x_0 = x, z = x_1 \sim \mathcal N(0, I)$을 상정하므로, 단순 이름 바꾸기를 통해 $p(x|z; \theta) = p(x_0|x_1; \theta)$를 얻을 수 있고, 이는 VAE의 generation process와 같다.

둘의 마지막 차이는 step 수의 차이 뿐이다.

**More step is better**

TBD

**Is diffusion model a 1,000-step VAE ?**

TBD

**References**

- Consistency Models, Song et al., 2023. [[arXiv:2303.01469](https://arxiv.org/abs/2303.01469)]
- VDM: Variational Diffusion Models, Kingma et al., 2021. [[arXiv:2107.00630](https://arxiv.org/abs/2107.00630)]
- NVAE: A Deep Hierarchical Variational Autoencoder, Vahdat & Kautz, 2020. [[arXiv:2007.03898](https://arxiv.org/abs/2007.03898)]
- DDPM: Denoising Diffusion Probabilistic Models, Ho et al., 2020. [[arXiv:2006.11239](https://arxiv.org/abs/2006.11239)]
- StyleGAN: A Style-Based Generator Architecture for Generative Adversarial Networks, Karras et al., 2018. [[arXiv:1812.04948](https://arxiv.org/abs/1812.04948)]
- GAN: Generative Adversarial Networks, Goodfellow et al., 2014. [[arXiv:1406.2661](https://arxiv.org/abs/1406.2661)]
- VAE: Autoencoding Variational Bayes, Kingma & Welling, 2013. [[arXiv:1312.6114](https://arxiv.org/abs/1312.6114)]
