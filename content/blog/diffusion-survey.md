---
title: "[WIP] Diffusion, Flow Survey"
date: 2025-02-09T13:09:43+09:00
draft: false

# post thumb
image: "images/post/diffusion-survey/head.png"

# meta description
description: "Diffusion, Flow Survey"

# taxonomies
categories:
  - "Generative"
tags:
  - "Machine Learning"
  - "Deep Learning"
  - "Generative"
  - "Bayesian"
  - "DDPM"
  - "Denoising Diffusion"
  - "Diffusion"
  - "Consistency model"
  - "Normalizing flow"
  - "Stochastic Process"
  - "VAE"
  - "Likelihood"
  - "Oksendal"
  - "SDE"

# post type
type: "post"
---

- Survey of Diffusion, Flow Models
- Keyword: Bayesian, VAE, Diffusion Models, Score Models, Schrodinger Bridge, Normalizing Flows, Rectified Flows, Neural ODE, Consistency Models

**Abstract**

2013년 VAE[[Kingma & Welling, 2013.](https://arxiv.org/abs/1312.6114)], 2014년 GAN[[Goodfellow et al., 2014.](https://arxiv.org/abs/1406.2661)]을 지나 2020년의 DDPM[[Ho et al., 2020.](https://arxiv.org/abs/2006.11239)]과 2022년의 Flow Matching[[Lipman et al., 2022.](https://arxiv.org/abs/2210.02747)]까지, 생성 모델은 다양한 형태로 발전해 왔다. 기존까지의 생성 모델을 짚어보고, 앞으로의 방향성에 관하여 고민해 보고자 한다.

**Introduction**

Supervised Learning은 흔히 입력 데이터 $X$와 출력 데이터 $Y$의 데이터셋 $D$가 주어진다; $(x, y)\in D$. 이때 데이터셋 $D$의 분포 $\Pi(X, Y)$를 X와 Y의 Coupling이라 정의하자; $(x, y)\sim\Pi(X, Y)$ \
(simply assume the pdf $p_{X,Y}$ of $\Pi(X, Y)$ as $p_{X, Y}(x, y) = \delta_{(x, y)\in D}$ for dirac-delta $\delta$ and $(x, y)\in X\times Y$)

많은 경우에 Supervised Learning은 parametrized function $f_\theta: X \to Y$를 통해 $x\mapsto y$의 대응을 학습하고, 대개 조건부 분포의 likelihood를 maximizing 하는 방식으로 이뤄진다.

$$\hat\theta = \arg\max_\theta \sum_{(x, y)\sim\Pi(X, Y)} \log p_{Y|X}(f_\theta(x)|x)$$

만약 조건부 분포를 정규 분포로 가정한다면, 이는 흔히 알려진 Mean Squared Error; MSE의 형태로 정리된다.

$$\log p_{Y|X}(f_\theta(x)|x) \propto -||f_\theta(x) - y||^2 + C \implies \hat\theta = \arg\min_\theta \sum_{(x, y)\sim\Pi(X, Y)}||f_\theta(x) - y||^2$$

생성 모델(Generative Model)은 주어진 데이터의 확률 분포 학습을 목적으로 한다. 이는 probability mass function; pmf, 혹은 probability density function; pdf를 데이터로부터 추정하거나, 데이터 분포의 표본을 생성하는 Generator를 학습하는 방식으로 이뤄진다.

데이터 $X$의 분포를 $\pi_X$라 할 때, $\pi_X$의 pdf $p_X(x)$를 학습하거나, known distribution(e.g. $\mathcal N(0, I)$)의 표본 $z\sim Z$를 데이터 분포의 한 점 $x'\sim\pi_X$으로 대응하는 Generator $G: Z \to X$를 학습한다.

이 경우 대부분 사전 분포와 데이터 분포의 Coupling은 독립으로 가정하여(i.e. $\Pi(Z, X) = \pi_Z\times \pi_X$), parameterized generator $G_\theta$에 대해 log-likelihood를 maximizing 하거나; $\max_\theta \log p_X(G_\theta(\cdot))$, 분포 간 거리를 측정할 수 있는 differentiable objective $D$를 두어 최적화하기도 한다; $\min_\theta \sum_{(x, z)\sim\Pi(Z, X)} D(G_\theta(z), x)$.

전자의 상황에서 Generator가 $z\sim Z$의 조건부 분포를 표현하는 것은 자명하다; $G_\theta(z)\sim p_{\theta, X|Z}(\cdot|z)$. $p_X$의 형태를 모를 때(혹은 가정하지 않을 때), 우리는 조건부 분포를 $Z$에 대해 marginalize 하여(i.e. $p_{\theta, X}$) 데이터셋 $X$에 대해 maximize 하는 선택을 할 수 있다; $\max_\theta \sum_{x\sim\pi_x}\log p_{\theta, X}(x)$

(후자는 GAN에 관한 논의로 이어지므로, 현재의 글에서는 다루지 않는다.)

조건부 분포를 marginalize 하기 위해서는 $p_{\theta,X}(x) = \int_Z p_Z(z)p_{\theta,X|Z}(x|z)dz$의 적분 과정이 필요한데, neural network로 표현된 $G_\theta$의 조건부 분포 $p_{\theta,X}$를 적분하는 것은 사실상 불가능하다(intractable).

만약 이를 $\Pi(X, Y)$에 대해 충분히 Random sampling 하여 Emperical average를 취하는 방식으로 근사한다면(i.e. Monte Carlo Estimation), 대형 데이터셋을 취급하는 현대의 문제 상황에서는 Resource Exhaustive 할 것이다. 특히나 Independent Coupling을 가정하고 있기에, Emperical Estimation의 분산이 커 학습에 어려움을 겪을 가능성이 높다. 분산을 줄이기 위해 표본을 늘린다면 컴퓨팅 리소스는 더욱더 많이 필요할 것이다.

현대의 생성 모델은 이러한 문제점을 다양한 관점에서 풀어 나간다. Invertible Generator를 두어 치환 적분(change-of-variables)의 형태로 적분 문제를 우회하기도 하고, 적분 없이 likelihood의 하한을 구해 maximizing lower bound의 형태로 근사하는 경우도 있다.

아래의 글에서는 2013년 VAE[[Kingma & Welling, 2013.](https://arxiv.org/abs/1312.6114)]부터 차례대로 각각의 생성 모델이 어떤 문제를 해결하고자 하였는지, 어떤 방식으로 해결하고자 하였는지 살펴보고자 한다. VAE[[Kingma & Welling, 2013.](https://arxiv.org/abs/1312.6114), [Vahdat & Kautz, 2020.](https://arxiv.org/abs/2007.03898)]를 시작으로, Normalizing Flows[[Rezende & Mahamed, 2015.](https://arxiv.org/abs/1505.05770), [Kingma & Dhariwal, 2018.](https://arxiv.org/abs/1807.03039)], Neural ODE[[Chen et al., 2018](https://arxiv.org/abs/1806.07366)], Score Models[[Song & Ermon, 2019.](https://arxiv.org/abs/1907.05600), [Song et al., 2020.](https://arxiv.org/abs/2011.13456)], Diffusion Models[[Ho et al., 2020.](https://arxiv.org/abs/2006.11239), [Song et al., 2020.](https://arxiv.org/abs/2010.02502)], Flow Matching[[Liu et al., 2022.](https://arxiv.org/abs/2209.03003), [Lipman et al., 2022.](https://arxiv.org/abs/2210.02747)], Consistency Models[[Song et al., 2023.](https://arxiv.org/abs/2303.01469,), [Lu & Song, 2024.](https://arxiv.org/abs/2410.11081)], Schrodinger Bridge[[Shi et al., 2023.](https://arxiv.org/abs/2303.16852)]에 관해 이야기 나눠본다.

**VAE: Varitational Autoencoder**

- VAE: Auto-Encoding Variational Bayes, Kingma & Welling, 2013. [[arXiv:1312.6114](https://arxiv.org/abs/1312.6114)]

2013년 Kingma와 Welling은 VAE를 발표한다.

**References**

- VAE: Auto-Encoding Variational Bayes, Kingma & Welling, 2013. [[arXiv:1312.6114](https://arxiv.org/abs/1312.6114)]
- GAN: Generative Adversarial Networks, Goodfellow et al., 2014. [[arXiv:1406.2661](https://arxiv.org/abs/1406.2661)]
- DDPM: Denoising Diffusion Probabilistic Models, Ho et al., 2020. [[arXiv:2006.11239](https://arxiv.org/abs/2006.11239)]
- Flow Matching for Generative Modeling, Lipman et al., 2022. [[arXiv:2210.02747](https://arxiv.org/abs/2210.02747)]
- NVAE: A Deep Hierarchical Variational Autoencoder, Vahdat & Kautz, 2020. [[arXiv:2007.03898](https://arxiv.org/abs/2007.03898)]
- Variational Inference with Normalizing Flows , Rezende & Mahamed, 2015. [[arXiv:1505.05770](https://arxiv.org/abs/1505.05770)]
- Glow: Generative Flow and Invertible 1x1 Convolutions, Kingma & Dhariwal, 2018. [[arXiv:1807.03039](https://arxiv.org/abs/1807.03039)]
- NODE: Neural Ordinary Differential Equations, Chen et al., 2018. [[arXiv:1806.07366](https://arxiv.org/abs/1806.07366)]
- NCSN: Generative Modeling by Estimating Gradients of the Data Distribution, Song & Ermon, 2019. [[arXiv:1907.05600](https://arxiv.org/abs/1907.05600)]
- Score-Based Generative Modeling through Stochastic Differential Equations, Song et al., 2020. [[arXiv:2011.13456](https://arxiv.org/abs/2011.13456)]
- DDPM: Denoising Diffusion Probabilistic Models, Ho et al., 2020. [[arXiv:2006.11239](https://arxiv.org/abs/2006.11239)]
- DDIM: Denoising Diffusion Implicit Models, Song et al., 2020. [[arXiv:2010.02502](https://arxiv.org/abs/2010.02502)]
- Rectified Flow: Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow, Liu et al., 2022. [[arXiv:2209.03003](https://arxiv.org/abs/2209.03003)]
- Flow Matching for Generative Modeling, Lipman et al., 2022. [[arXiv:2210.02747](https://arxiv.org/abs/2210.02747)]
- Consistency Models, Song et al., 2023. [[arXiv:2303.01469](https://arxiv.org/abs/2303.01469)]
- Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models, Lu & Song, 2024. [[arXiv:2410.11081](https://arxiv.org/abs/2410.11081)]
- DSBM: Diffusion Schrodinger Bridge Matching, Shi et al., 2023. [[arXiv:2303.16852](https://arxiv.org/abs/2303.16852)]

---

{{< details summary="TODO" >}}

0. Preliminaries

Oksendal SDE
- Brownian Motion Model
- Ito process
- Ito Diffusion, Markovian Property

Normalizing Flows
- Variational Inference with Normalizing Flows
, Rezende & Mahamed, 2015. https://arxiv.org/abs/1505.05770, https://revsic.github.io/blog/realnvp/
- Glow: Generative Flow and Invertible 1x1 Convolutions, Kingma & Dhariwal, 2018. https://arxiv.org/abs/1807.03039, https://revsic.github.io/blog/glowflowpp/
- Neural Spline Flows, Durkan et al., https://arxiv.org/abs/1906.04032
- Augmented Normalizing Flows: Bridging the Gap Between Generative Flows and Latent Variable Models, Huang et al., 2020. https://arxiv.org/abs/2002.07101, https://revsic.github.io/blog/anfvf/

Neural ODE
- Invertible Residual Networks, Behrmann et al., 2018. https://arxiv.org/abs/1811.00995, https://revsic.github.io/blog/resflow/
- Neural Ordinary Differential Equations, Chen et al., 2018. https://arxiv.org/abs/1806.07366

1. Score model
- Generative Modeling by Estimating Gradients of the Data Distribution, Song & Ermon, https://arxiv.org/abs/1907.05600
- Score-Based Generative Modeling through Stochastic Differential Equations, Song et al., https://arxiv.org/abs/2011.13456

2. DDPM
- Denoising Diffusion Probabilistic Models, Ho et al., 2020. https://arxiv.org/abs/2006.11239, https://revsic.github.io/blog/diffusion/
- Diffusion Models Beat GANs on Image Synthesis, Dhariwal & Nichol, 2021. https://arxiv.org/abs/2105.05233
- Variational Diffusion Models, Kingma et al., 2021. https://arxiv.org/abs/2107.00630, https://revsic.github.io/blog/vdm/
- Denoising Diffusion Implicit Models, Song et al., 2020. https://arxiv.org/abs/2010.02502
- Classifier-Free Diffusion Guidance, Ho & Salimans, 2022. https://arxiv.org/abs/2207.12598
- [Blog] Essay: VAE as a 1-step Diffusion Model
, https://revsic.github.io/blog/1-step-diffusion/

3. SDE & PF ODE
- Score-Based Generative Modeling through Stochastic Differential Equations, Song et al., 2020. https://arxiv.org/abs/2011.13456

4. Rectified Flow & Flow Matching
- Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow, Liu et al., 2022. https://arxiv.org/abs/2209.03003
- Flow Matching for Generative Modeling, Lipman et al., 2022. https://arxiv.org/abs/2210.02747
- Simple ReFlow: Improved Techniques for Fast Flow Models, Kim et al., 2024. https://arxiv.org/abs/2410.07815s
- Improving the Training of Rectified Flows, Lee et al., 2024. https://arxiv.org/abs/2405.20320

5. Consistency Models
- Consistency Models, Song et al., 2023. https://arxiv.org/abs/2303.01469, https://revsic.github.io/blog/cm/
- Inconsistencies In Consistency Models: Better ODE Solving Does Not Imply Better Samples, Vouitsis et al., 2024. https://arxiv.org/abs/2411.08954
- Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models, Lu & Song, 2024. https://arxiv.org/abs/2410.11081

6. Bridge
- Diffusion Schrodinger Bridge Matching, Shi et al., 2023. https://arxiv.org/abs/2303.16852

7. Furthers
Unified view
- SurVAE Flows: Surjections to Bridge the Gap between VAEs and Flows, Nielsen et al., 2020. https://arxiv.org/abs/2007.02731, https://revsic.github.io/blog/survaeflow/
- Simulation-Free Training of Neural ODEs on Paired Data, Kim et al., 2024. https://arxiv.org/abs/2410.22918
- Simulation-Free Differential Dynamics through Neural Conservation Laws, Hua et al., ICLR 2025. https://openreview.net/forum?id=jIOBhZO1ax

Fewer-step approaches
- Progressive Distillation for Fast Sampling of Diffusion Models, Salimans & Ho, 2022. https://arxiv.org/abs/2202.00512
- Tackling the Generative Learning Trilemma with Denoising Diffusion GANs, Xiao et al., 2021.
- InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation, Liu et al., 2023. https://arxiv.org/abs/2309.06380
- One Step Diffusion via Shortcut Models, Frans et al,. 2024. https://arxiv.org/abs/2410.12557

Velocity consistency
- Rectified Diffusion: Straightness Is Not Your Need in Rectified Flow, Want et al., 2024. https://arxiv.org/abs/2410.07303
- Consistency Flow Matching: Defining Straight Flows with Velocity Consistency, Yang et al., 2024. https://arxiv.org/abs/2407.02398

- [Blog] Essay: Generative models, Mode coverage, https://revsic.github.io/blog/coverage/

{{</details>}}
