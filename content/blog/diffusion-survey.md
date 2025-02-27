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

Supervised Learning은 흔히 입력 데이터 $X$와 출력 데이터 $Y$가 주어진다; $(x, y)\in D$. 이때 데이터셋 $D$의 분포 $\Pi(X, Y)$를 X와 Y의 Coupling이라 정의하자; $(x, y)\sim\Pi(X, Y)$ \
(e.g. the pdf $p_{X,Y}$ of $\Pi(X, Y)$ as $p_{X, Y}(x, y) = \delta_{(x, y)\in D}$ for dirac-delta $\delta$ and $(x, y)\in X\times Y$)

많은 경우에 Supervised Learning은 parametrized function $f_\theta: X \to Y$를 통해 $x\mapsto y$의 대응을 학습하고, 조건부 분포의 likelihood를 maximizing 하는 방식으로 이뤄진다.

$$\hat\theta = \arg\max_\theta \sum_{(x, y)\sim\Pi(X, Y)} \log p_{Y|X}(f_\theta(x)|x)$$

만약 조건부 분포를 정규 분포로 가정한다면, 이는 흔히 알려진 Mean Squared Error; MSE의 형태로 정리된다.

$$\log p_{Y|X}(f_\theta(x)|x) \propto -||f_\theta(x) - y||^2 + C \implies \hat\theta = \arg\min_\theta \sum_{(x, y)\sim\Pi(X, Y)}||f_\theta(x) - y||^2$$

생성 모델(Generative Model)은 주어진 데이터의 확률 분포 학습을 목적으로 한다. 이는 probability mass function; pmf, 혹은 probability density function; pdf를 데이터로부터 추정하거나, 데이터 분포의 표본을 생성하는 Generator를 학습하는 방식으로 이뤄진다.

데이터 $X$의 분포를 $\pi_X$라 할 때, $\pi_X$의 pdf $p_X(x)$를 학습하거나, known distribution(e.g. $\mathcal N(0, I)$)의 표본 $z\sim Z$를 데이터 분포의 한 점 $x'\sim\pi_X$으로 대응하는 Generator $G: Z \to X$를 학습한다.

이 경우 대부분 사전 분포와 데이터 분포의 Coupling은 독립으로 가정하여(i.e. $\Pi(Z, X) = \pi_Z\times \pi_X$), parameterized generator $G_\theta$에 대해 log-likelihood를 maximizing 하거나; $\max_\theta \log p_X(G_\theta(\cdot))$, 분포 간 거리를 측정할 수 있는 differentiable objective $D$를 두어 최적화하기도 한다; $\min_\theta \sum_{(x, z)\sim\Pi(Z, X)} D(G_\theta(z), x)$.

전자의 상황에서 Generator가 $z\sim Z$의 조건부 분포를 표현하는 것은 자명하다; $G_\theta(z)\sim p_{\theta, X|Z}(\cdot|z)$. 우리는 $p_X$의 형태를 모를 때(혹은 가정하지 않을 때), 조건부 분포를 $Z$에 대해 marginalize 하여(i.e. $p_{\theta, X}$) 데이터셋 $X$에 대해 maximize 하는 선택을 할 수 있다; $\max_\theta \sum_{x\sim\pi_X}\log p_{\theta, X}(x)$

(후자는 GAN에 관한 논의로 이어지므로, 현재의 글에서는 다루지 않는다.)

조건부 분포를 marginalize 하기 위해서는 $p_{\theta,X}(x) = \int_Z p_Z(z)p_{\theta,X|Z}(x|z)dz$의 적분 과정이 필요한데, neural network로 표현된 $G_\theta$의 조건부 분포 $p_{\theta,X}$를 적분하는 것은 사실상 불가능하다(intractable).

만약 이를 $\Pi(X, Y)$에 대해 충분히 Random sampling 하여 Emperical average를 취하는 방식으로 근사한다면(i.e. Monte Carlo Estimation), 대형 데이터셋을 취급하는 현대의 문제 상황에서는 Resource Exhaustive 할 것이다. 특히나 Independent Coupling을 가정하고 있기에, Emperical Estimation의 분산이 커 학습에 어려움을 겪을 가능성이 높다. 분산을 줄이기 위해 표본을 늘린다면 컴퓨팅 리소스는 더욱더 많이 필요할 것이다.

현대의 생성 모델은 이러한 문제점을 다양한 관점에서 풀어 나간다. Invertible Generator를 두어 치환 적분(change-of-variables)의 형태로 적분 문제를 우회하기도 하고, 적분 없이 likelihood의 하한을 구해 maximizing lower bound의 형태로 근사하는 경우도 있다.

아래의 글에서는 2013년 VAE[[Kingma & Welling, 2013.](https://arxiv.org/abs/1312.6114)]부터 차례대로 각각의 생성 모델이 어떤 문제를 해결하고자 하였는지, 어떤 방식으로 해결하고자 하였는지 살펴보고자 한다. VAE[[Kingma & Welling, 2013.](https://arxiv.org/abs/1312.6114), [NVAE; Vahdat & Kautz, 2020.](https://arxiv.org/abs/2007.03898)]를 시작으로, Normalizing Flows[[RealNVP; Dinh et al., 2016.](https://arxiv.org/abs/1605.08803), [Glow; Kingma & Dhariwal, 2018.](https://arxiv.org/abs/1807.03039)], Neural ODE[[NODE; Chen et al., 2018](https://arxiv.org/abs/1806.07366)], Score Models[[NCSN; Song & Ermon, 2019.](https://arxiv.org/abs/1907.05600), [Song et al., 2020.](https://arxiv.org/abs/2011.13456)], Diffusion Models[[DDPM; Ho et al., 2020.](https://arxiv.org/abs/2006.11239), [DDIM; Song et al., 2020.](https://arxiv.org/abs/2010.02502)], Flow Matching[[Liu et al., 2022.](https://arxiv.org/abs/2209.03003), [Lipman et al., 2022.](https://arxiv.org/abs/2210.02747)], Consistency Models[[Song et al., 2023.](https://arxiv.org/abs/2303.01469,), [Lu & Song, 2024.](https://arxiv.org/abs/2410.11081)], Schrodinger Bridge[[DSBM; Shi et al., 2023.](https://arxiv.org/abs/2303.16852)]에 관해 이야기 나눠본다.

**VAE: Variational Autoencoder**

- VAE: Auto-Encoding Variational Bayes, Kingma & Welling, 2013. [[arXiv:1312.6114](https://arxiv.org/abs/1312.6114)]

2013년 Kingma와 Welling은 VAE를 발표한다. VAE의 시작점은 위의 Introduction과 같다. Marginalize 과정은 intractable하고, Monte Carlo Estimation을 하기에는 컴퓨팅 자원이 과요구된다.

이에 VAE는 $z$의 intractable posterior $p_{Z|X}(z|x) = p_{Z, X}(z, x)/p_X(x)$를 Neural network $E_\phi(x)\sim p_{\phi,Z|X}(\cdot|x)$ 로 대치하는 방식을 택하고, 이를 approximate posterior $q_\phi(z|x) = p_{\phi,Z|X}(z|x)$로 표기한다.

$$\begin{align*}
\log p_{\theta, X}(x) &= \mathbb E_{z\sim q_\phi(\cdot|x)} \log p_{\theta, X}(x) \\\\
&= \mathbb E_{z\sim q_\phi(\cdot|x)}\left[\log p_{\theta, X}(x) + \log\frac{p_{\theta,Z,X}(z, x)q_\phi(z|x)}{p_{\theta,Z,X}(z, x)q_\phi(z|x)}\right] \\\\
&= \mathbb E_{z\sim q_\phi(\cdot|x)}\left[\log\frac{p_Z(z)p_{\theta,X|Z}(x|z)\cdot q_\phi(z|x)}{p_{\theta,Z|X}(z|x)\cdot q_\phi(z|x)} \right] \\\\
&= \mathbb E_{z\sim q_\phi(\cdot|x)}\left[\log\frac{q_\phi(z|x)}{p_{\theta,Z|X}(z|x)} - \log\frac{q_\phi(z|x)}{p_Z(z)} + \log p_{\theta,X|Z}(x|z)\right] \\\\
&= D_{KL}(q_\phi(z|x)||p_{\theta,Z|X}(z|x)) - D_{KL}(q_\phi(z|x)||p_Z(z)) + \mathbb E_{z\sim q_\phi(\cdot|x)}\log p_{\theta,X|Z}(x|z)
\end{align*}$$

$q_\phi(z|x)$의 도입과 함께 $\log p_{\theta, X}(x)$는 위와 같이 정리된다. 순서대로 $D_{KL}(q_\phi(z|x)||p_{\theta,Z|X}(z|x))$은 approximate posterior와 true posterior의 KL-Divergence, $D_{KL}(q_\phi(z|x)||p_{Z}(z))$는 사전 분포 $p_Z(z)$와의 divergence, $\mathbb E_{z\sim q_\phi(\cdot|x)}\log p_{\theta, X|Z}(x|z)$는 reconstruction을 다루게 된다.

여기서 계산이 불가능한 true posterior $p_{\theta, Z|X}(z|x)$를 포함한 항을 제외하면, 다음의 하한을 얻을 수 있으며 이를 Evidence Lower Bound라 한다(이하 ELBO). VAE는 ELBO $\mathcal L_{\theta, \phi}$를 Maximize 하는 방식으로 확률 분포를 학습한다.

$$\log p_{\theta, X}(x)\ge \mathbb E_{z\sim q_\phi(\cdot|x)}\log p_{\theta, X|Z}(x|z)- D_{KL}(q_\phi(z|x)||p_Z(z)) = \mathcal L_{\theta, \phi}(x)\ \ (\because D_{KL} \ge 0)$$

ELBO를 maximize하는 과정은 approximate posterior가 사전 분포와의 관계성을 유지하면서도, 데이터를 충분히 결정지을 수 있길 바라는 것이다.

이 과정은 Expectation 내에 $z\sim q_\phi(\cdot|x)$의 Sampling을 상정하고 있지만, Sampling 자체는 미분을 지원하지 않아 Gradient 기반의 업데이트를 수행할 수 없다. VAE는 이를 우회하고자, approximate posterior의 분포를 Gaussian으로 가정한다(i.e. $z\sim \mathcal N(\mu_\phi(x), \sigma_\phi^2(x)I)$).

$z = \mu_\phi(x) + \sigma_\phi(x)\zeta;\ \zeta\sim \mathcal N(0, I)$로 표본 추출을 대치하여 $E_\phi = (\mu_\phi, \sigma_\phi)$ 역시 학습할 수 있도록 두었다(i.e. reparametrization trick). 이때 $z_i\sim\mathcal N(\mu_\phi(x), \sigma^2_\phi(x)I)$를 몇 번 샘플링하여 평균을 구할 것인지 실험하였을 때(i.e. $1/N\cdot \sum_i^N\log p(x|z_i)$), 학습의 Batch size가 커지면 각 1개 표본만을 활용해도(N=1) 무방했다고 한다.

```py {style=github}
mu, sigma = E_phi(x)
# reparametrization
z = mu + sigma * torch.randn(...)
# ELBO
loss = (
  # log p(x|z)
  (x - G_theta(z)).square().mean()
  # log p(z)
  + z.square().mean()
  # - log q(z|x)
  - ((z - mu) / sigma).square().mean()
)
```

VAE는 Approximate posterior를 도입하여 Intractable likelihood를  근사하는 방향으로 접근하였고, Posterior 기반 Coupling을 통해 분산을 줄여 Monte Carlo Estimation의 시행 수를 줄일 수 있었다.

하지만 VAE 역시 여러 한계를 보였다.

$D_{KL}(q_\phi(z|x)||p_Z(z))$의 수렴 속도가 다른 항에 비해 상대적으로 빨라 posterior가 reconstruction에 필요한 정보를 충분히 담지 못하였고, 이는 Generator의 성능에 영향을 미쳤다. 이에 KL-Annealing/Warmup 등의 다양한 엔지니어링 기법이 소개되기도 한다.

또한, 뒤에 소개될 Normalizing Flows, Diffusion Models, GAN에 비해 Sample이 다소 Blurry 하는 등 품질이 높지 않았다. 이에는 Reconstruction loss가 MSE의 형태이기에 Blurry 해진다는 이야기, Latent variable의 dimension이 작아 그렇다는 이야기, 구조적으로 Diffusion에 비해 NLL이 높을 수밖에 없다는 논의 등 다양한 이야기가 뒤따랐다.

이에 VAE의 성능 개선을 위해 노력했던 연구 중, NVIDIA의 NVAE 연구를 소개하고자 한다.

---

- NVAE: A Deep Hierarchical Variational Autoencoder, Vahdat & Kautz, NeurIPS 2020. [[arXiv:2007.03898](https://arxiv.org/abs/2007.03898)]

NVAE(Nouveau VAE)는 프랑스어 `Nouveau: 새로운`의 뜻을 담아 *make VAEs great again*을 목표로 한다.

당시 VAE는 네트워크를 더 깊게 가져가고, Latent variable $z$를 단일 벡터가 아닌 여럿 두는 등(e.g. $z = \\{z_1, ..., z_N\\}$) Architectural Scaling에 초점을 맞추고 있었다[[VDVAE; Child, 2020.](https://arxiv.org/abs/2011.10650)]. 특히나 StyleGAN[[Karras et al., 2018.](https://arxiv.org/abs/1812.04948), [Karras et al., 2019.](https://arxiv.org/abs/1912.04958)], DDPM[[Ho et al., 2020.](https://arxiv.org/abs/2006.11239)] 등의 생성 모델이 Latent variable의 크기를 키우며 성능을 확보해 나가는 당대 분위기상 VAE에서도 유사한 시도가 여럿 보였다(관련 블로그: [Essay: VAE as a 1-step Diffusion Model](/blog/1-step-diffusion)).

{{< figure src="/images/post/diffusion-survey/nvae.png" width="60%" caption="Figure 2: The neural networks implementing an encoder and generative model. (Vahdat & Kautz, 2020)" >}}

NVAE는 latent groups $z = \\{z_1, z_2, ... z_L\\}$에 대해 $q(z|x) = \Pi_l q(z_l|z_{<1}, x)$의 hierarchical approximate posterior를 활용한다. ELBO는 다음과 같다.

$$\mathcal L_{VAE}(x) = \mathbb E_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z_1|x)||p(z_1)) - \sum^L_{l=2}\mathbb E_{q(z_{<l}|x)}[D_{KL}(q(z_l|x, z_{<l})||p(z_l))]$$

Encoder가 이미지로부터 feature map `r`를 생성(i.e. hierarchical approximate posterior, $q(z_l|x, z_{<l})$), Decoder가 trainable basis `h`로부터 Encoder feature map을 역순으로 더해가며 이미지를 생성하는 U-Net 구조를 상상하자. Generation 단계에서는 Encoder feature map `r`이 주어지지 않기에, feature map의 prior distribution $p(z_l)$의 샘플로 대체한다. 이는 어찌 보면 Spatial noise를 더해가는 StyleGAN[[Karras et al., 2018.](https://arxiv.org/abs/1812.04948)]과도 형태가 유사하다.

다만 이렇게 될 경우, $D_{KL}$의 조기 수렴에 따라 posterior collapse가 발생할 가능성이 높기에, 여러 engineering trick이 함께 제안되었다. Decoder에는 Depthwise-seperable convolution을 활용하지만 Encoder에서는 사용하지 않고, SE Block[[Hu et al., 2017.](https://arxiv.org/abs/1709.01507)]과 Spectral regularization, KL Warmup 도입, Batch normalization의 momentum parameter 조정 등이 있다.

이를 통해 실제로 당시 Normalizing Flows와 VAE 계열 모델 중에서는 좋은 성능을 보였다. 하지만 논문에서는 NLL(bit/dim)에 관한 지표만 보일 뿐, FID나 Precision/Recall 등 지표는 보이지 않아 다른 모델과의 비교는 쉽지 않았다.

정성적으로 보았을 때는 NVAE는 여전히 다소 Blurry 한 이미지를 보이거나, 인체의 형태가 왜곡되는 등의 Degenerate Mode가 관찰되는 등 아쉬운 모습을 보이기도 했다.

---

**Normalizing Flows**

- RealNVP: Density estimation using Real NVP, Dinh et al., 2016. [[arXiv:1605.08803](https://arxiv.org/abs/1605.08803)]

TBD

---

- ANF: Augmented Normalizing Flows: Bridging the Gap Between Generative Flows and Latent Variable Models, Huang et al., 2020. [[arXiv:2002.07101](https://arxiv.org/abs/2002.07101)]

TBD

---

- VFlow: More Expressive Generative Flows with Variational Data Augmentation, Chen et al., 2020. [[arXiv:2002.09741](https://arxiv.org/abs/2002.09741)]

TBD

---

- FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models, Grathwohl et al., 2018.  [[arXiv:1810.01367](https://arxiv.org/abs/1810.01367)]

TBD

**References**

- VAE: Auto-Encoding Variational Bayes, Kingma & Welling, 2013. [[arXiv:1312.6114](https://arxiv.org/abs/1312.6114)]
- GAN: Generative Adversarial Networks, Goodfellow et al., 2014. [[arXiv:1406.2661](https://arxiv.org/abs/1406.2661)]
- DDPM: Denoising Diffusion Probabilistic Models, Ho et al., 2020. [[arXiv:2006.11239](https://arxiv.org/abs/2006.11239)]
- Flow Matching for Generative Modeling, Lipman et al., 2022. [[arXiv:2210.02747](https://arxiv.org/abs/2210.02747)]
- NVAE: A Deep Hierarchical Variational Autoencoder, Vahdat & Kautz, 2020. [[arXiv:2007.03898](https://arxiv.org/abs/2007.03898)]
- RealNVP: Density estimation using Real NVP, Dinh et al., 2016. [[arXiv:1605.08803](https://arxiv.org/abs/1605.08803)]
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
- VDVAE: Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images, Child, 2020. [[arXiv:2011.10650](https://arxiv.org/abs/2011.10650)]
- StyleGAN: A Style-Based Generator Architecture for Generative Adversarial Networks, Karras et al., 2018. [[arXiv:1812.04948](https://arxiv.org/abs/1812.04948)]
- StyleGAN2: Analyzing and Improving the Image Quality of StyleGAN, Karras et al., 2019. [[arXiv:1912.04958](https://arxiv.org/abs/1912.04958)]
- Squeeze-and-Excitation Networks, Hu et al., 2017. [[arXiv:1709.01507](https://arxiv.org/abs/1709.01507)]
- ANF: Augmented Normalizing Flows: Bridging the Gap Between Generative Flows and Latent Variable Models, Huang et al., 2020. [[arXiv:2002.07101](https://arxiv.org/abs/2002.07101)]
- VFlow: More Expressive Generative Flows with Variational Data Augmentation, Chen et al., 2020. [[arXiv:2002.09741](https://arxiv.org/abs/2002.09741)]
- FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models, Grathwohl et al., 2018.  [[arXiv:1810.01367](https://arxiv.org/abs/1810.01367)]

---

{{< details summary="TODO" >}}

0. Preliminaries

Oksendal SDE
- Brownian Motion Model
- Ito process
- Ito Diffusion, Markovian Property

Neural ODE
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
