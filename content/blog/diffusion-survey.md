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

Supervised Learning에서는 흔히 입력 데이터 $x\in X$와 출력 데이터 $y\in Y$를 가정한다. 이때 데이터셋 $D = \\{(x, y)\\}$의 분포 $\Pi(X, Y)$를 X와 Y의 Coupling이라 정의하자(i.e. $(x, y)\sim\Pi(X, Y)$). 단순히는 dirac delta $\delta$에 대해 $\Pi(X, Y)$의 pdf를 $p_{X, Y}(x, y) = \delta_{(x, y)\in D}; (x, y)\in (X, Y)$로 가정해볼 수 있다.

많은 경우에 Supervised Learning은 parametrized function $f_\theta: X \to Y$를 통해 $x\mapsto y$의 대응을 학습하고, 조건부 분포의 likelihood를 maximizing 하는 방식으로 이뤄진다.

$$\hat\theta = \arg\max_\theta \sum_{(x, y)\sim\Pi(X, Y)} \log p_{Y|X}(f_\theta(x)|x)$$

만약 조건부 분포를 정규 분포로 가정한다면, 이는 흔히 알려진 Mean Squared Error의 형태로 정리된다.

$$\log p_{Y|X}(f_\theta(x)|x) \propto -||f_\theta(x) - y||^2 + C \implies \hat\theta = \arg\min_\theta \sum_{(x, y)\sim\Pi(X, Y)}||f_\theta(x) - y||^2$$

생성 모델(Generative Model)은 주어진 데이터의 확률 분포 학습을 목표로 한다. 이는 데이터로부터 probability density function을 추정하거나(혹은 probability mass function), Generator의 학습을 통해 데이터 분포의 표본을 생성하고자 한다.

i.e. 데이터 $X$의 분포를 $\pi_X$라 할 때, $\pi_X$의 pdf $p_X(x)$를 construct 하거나, known distribution(e.g. $\mathcal N(0, I)$)의 표본 $z\sim Z$를 데이터 분포의 한 점 $x'\sim\pi_X$으로 대응하는 Generator $G: Z \to X$를 학습한다.

이 경우 대부분 사전 분포와 데이터 분포의 Coupling은 독립으로 가정하며(i.e. $\Pi(Z, X) = \pi_Z\times \pi_X$), parameterized generator $G_\theta$에 대해 log-likelihood(i.e. $\log p_X(x)$)를 maximizing 하거나, 분포 간 거리를 측정할 수 있는 differentiable objective $D$를 최적화하기도 한다(i.e. $\min_\theta \sum_{(x, z)\sim\Pi(Z, X)} D(G_\theta(z), x)$).

Generator가 $z\sim Z$의 조건부 분포를 표현하는 것은 자명하다(i.e. $G_\theta(z)\sim p_{\theta, X|Z}(\cdot|z)$). 전자의 상황에서 우리는 $p_X$의 형태를 모를 때(혹은 가정하지 않을 때), 조건부 분포를 $Z$에 대해 marginalize 하여(i.e. $p_{\theta, X}$) 데이터셋 $X$에 대해 maximize 하는 선택을 할 수 있다. $\max_\theta \sum_{x\sim\pi_X}\log p_{\theta, X}(x)$

(후자는 GAN에 관한 논의로 이어지므로, 현재의 글에서는 다루지 않는다.)

조건부 분포를 marginalize 하기 위해서는 $p_{\theta,X}(x) = \int_Z p_Z(z)p_{\theta,X|Z}(x|z)dz$의 적분 과정이 필요한데, neural network로 표현된 $G_\theta$의 조건부 분포 $p_{\theta,X}$를 적분하는 것은 사실상 불가능하다(intractable).

만약 이를 $\Pi(X, Y)$에 대해 충분히 Random sampling 하여 Emperical average를 취하는 방식으로 근사한다면(i.e. Monte Carlo Estimation), 대형 데이터셋을 취급하는 현대의 문제 상황에서는 Resource Exhaustive 할 것이다. 특히나 Independent Coupling을 가정하고 있기에, Emperical Estimation의 분산이 커 학습에 어려움을 겪을 가능성이 높다. 분산을 줄이기 위해 표본을 늘린다면 컴퓨팅 리소스는 더욱더 많이 필요할 것이다.

현대의 생성 모델은 이러한 문제점을 다양한 관점에서 풀어 나간다. Invertible Generator를 두어 변수 치환(change-of-variables)의 형태로 적분 문제를 우회하기도 하고, 적분 없이 likelihood의 하한을 구해 maximizing lower bound의 형태로 근사하는 경우도 있다.

아래의 글에서는 2013년 VAE[[Kingma & Welling, 2013.](https://arxiv.org/abs/1312.6114)]부터 차례대로 각각의 생성 모델이 어떤 문제를 해결하고자 하였는지, 어떤 방식으로 해결하고자 하였는지 살펴보고자 한다. VAE[[Kingma & Welling, 2013.](https://arxiv.org/abs/1312.6114), [NVAE; Vahdat & Kautz, 2020.](https://arxiv.org/abs/2007.03898)]를 시작으로, Normalizing Flows[[RealNVP; Dinh et al., 2016.](https://arxiv.org/abs/1605.08803), [Glow; Kingma & Dhariwal, 2018.](https://arxiv.org/abs/1807.03039)], Neural ODE[[NODE; Chen et al., 2018](https://arxiv.org/abs/1806.07366)], Score Models[[NCSN; Song & Ermon, 2019.](https://arxiv.org/abs/1907.05600), [Song et al., 2020.](https://arxiv.org/abs/2011.13456)], Diffusion Models[[DDPM; Ho et al., 2020.](https://arxiv.org/abs/2006.11239), [DDIM; Song et al., 2020.](https://arxiv.org/abs/2010.02502)], Flow Matching[[Liu et al., 2022.](https://arxiv.org/abs/2209.03003), [Lipman et al., 2022.](https://arxiv.org/abs/2210.02747)], Consistency Models[[Song et al., 2023.](https://arxiv.org/abs/2303.01469,), [Lu & Song, 2024.](https://arxiv.org/abs/2410.11081)], Schrodinger Bridge[[DSBM; Shi et al., 2023.](https://arxiv.org/abs/2303.16852)]에 관해 이야기 나눠본다.

---

**VAE: Variational Autoencoder**

- VAE: Auto-Encoding Variational Bayes, Kingma & Welling, 2013. [[arXiv:1312.6114](https://arxiv.org/abs/1312.6114)]

2013년 Kingma와 Welling은 VAE를 발표한다. VAE의 시작점은 위의 Introduction과 같다. Marginalize 과정은 intractable하고, Monte Carlo Estimation을 하기에는 컴퓨팅 자원이 과요구된다.

이에 VAE는 $z$의 intractable posterior $p_{Z|X}(z|x) = p_{Z, X}(z, x)/p_X(x)$를 approximate posterior $E_\phi(x)\sim p_{\phi,Z|X}(\cdot|x)$ 로 대치하는 방식을 택한다. (아래는 편의를 위해 $q_\phi(z|x) = p_{\phi,Z|X}(z|x)$로 표기한다.)

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

이 과정은 Expectation 내에 $z\sim q_\phi(\cdot|x)$의 Sampling을 상정하고 있지만, Sampling 자체는 미분을 지원하지 않아 Gradient 기반의 업데이트를 수행할 수 없다. VAE는 이를 우회하고자, approximate posterior의 분포를 $z\sim \mathcal N(\mu_\phi(x), \sigma_\phi^2(x)I)$의 Gaussian으로 가정하고, $z = \mu_\phi(x) + \sigma_\phi(x)\zeta;\ \zeta\sim \mathcal N(0, I)$로 표본 추출을 대치하여 $E_\phi = (\mu_\phi, \sigma_\phi)$ 역시 학습할 수 있도록 두었다(i.e. reparametrization trick).

이때 $z_i\sim\mathcal N(\mu_\phi(x), \sigma^2_\phi(x)I)$를 몇 번 샘플링하여 평균을 구할 것인지 실험하였을 때(i.e. $1/N\cdot \sum_i^N\log p(x|z_i)$), 학습의 Batch size가 커지면(논문에서는 100개) 각 1개 표본만을 활용해도(N=1) 성능상 큰 차이가 없었다고 한다.

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

$D_{KL}(q_\phi(z|x)||p_Z(z))$의 수렴 속도가 다른 항에 비해 상대적으로 빨라 posterior가 reconstruction에 필요한 정보를 충분히 담지 못하였고, Generator의 성능에도 영향을 미쳤다. 이에 KL-Annealing/Warmup 등의 다양한 엔지니어링 기법이 소개되기도 한다.

또한, 뒤에 소개될 Normalizing Flows, Diffusion Models, GAN에 비해 Sample이 다소 Blurry 하는 등 품질이 높지 않았다. 이에는 Reconstruction loss가 MSE의 형태이기에 Blurry 해진다는 이야기, Latent variable의 dimension이 작아 그렇다는 이야기, 구조적으로 Diffusion에 비해 NLL이 높을 수밖에 없다는 논의 등 다양한 이야기가 뒤따랐다.

이에 VAE의 성능 개선을 위해 노력했던 연구 중, NVIDIA의 NVAE 연구를 소개하고자 한다.

---

- NVAE: A Deep Hierarchical Variational Autoencoder, Vahdat & Kautz, NeurIPS 2020. [[arXiv:2007.03898](https://arxiv.org/abs/2007.03898)]

NVAE(Nouveau VAE)는 프랑스어 `Nouveau: 새로운`의 뜻을 담아 *make VAEs great again*을 목표로 한다.

당시 VAE는 네트워크를 더 깊게 가져가고, Latent variable $z$를 단일 벡터가 아닌 여럿 두는 등(e.g. $z = \\{z_1, ..., z_N\\}$) Architectural Scaling에 초점을 맞추고 있었다(e.g. [VDVAE; Child, 2020.](https://arxiv.org/abs/2011.10650)). 특히나 StyleGAN[[Karras et al., 2018.](https://arxiv.org/abs/1812.04948), [Karras et al., 2019.](https://arxiv.org/abs/1912.04958)], DDPM[[Ho et al., 2020.](https://arxiv.org/abs/2006.11239)] 등의 생성 모델이 Latent variable의 크기를 키우며 성능을 확보해 나가는 당대 분위기상 VAE에서도 유사한 시도가 여럿 보였다[blog:[Essay: VAE as a 1-step Diffusion Model](/blog/1-step-diffusion)].

{{< figure src="/images/post/diffusion-survey/nvae.png" width="60%" caption="Figure 2: The neural networks implementing an encoder and generative model. (Vahdat & Kautz, 2020)" >}}

NVAE는 latent groups $z = \\{z_1, z_2, ... z_L\\}$에 대해 $q(z|x) = \Pi_l q(z_l|z_{<1}, x)$의 hierarchical approximate posterior를 활용한다. ELBO는 다음과 같다.

$$\mathcal L_{VAE}(x) = \mathbb E_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z_1|x)||p(z_1)) - \sum^L_{l=2}\mathbb E_{q(z_{<l}|x)}[D_{KL}(q(z_l|x, z_{<l})||p(z_l))]$$

Encoder가 이미지로부터 feature map `r`를 생성(i.e. hierarchical approximate posterior, $q(z_l|x, z_{<l})$), Decoder가 trainable basis `h`로부터 Encoder feature map을 역순으로 더해가며 이미지를 생성하는 U-Net 구조를 상상하자. Generation 단계에서는 Encoder feature map `r`이 주어지지 않기에, feature map의 prior distribution $p(z_l)$의 샘플로 대체한다. 이는 어찌 보면 Spatial noise를 더해가는 StyleGAN[[Karras et al., 2018.](https://arxiv.org/abs/1812.04948)]과도 형태가 유사하다.

다만 이 경우, $D_{KL}$의 조기 수렴에 따라 posterior collapse가 발생할 가능성이 높기에, 여러 engineering trick이 함께 제안되었다. Decoder에는 Depthwise-seperable convolution을 활용하지만 Encoder에서는 사용하지 않고, SE Block[[Hu et al., 2017.](https://arxiv.org/abs/1709.01507)]과 Spectral regularization, KL Warmup 도입, Batch normalization의 momentum parameter 조정 등이 있다.

이를 통해 실제로 당시 Normalizing Flows와 VAE 계열 모델 중에서는 좋은 성능을 보였다. 하지만 논문에서는 NLL(bit/dim)에 관한 지표만 보일 뿐, FID나 Precision/Recall 등 지표는 보이지 않아 다른 모델과의 비교는 쉽지 않았다.

정성적으로 보았을 때는 NVAE는 여전히 다소 Blurry 한 이미지를 보이거나, 인체의 형태가 종종 왜곡되는 등의 Degenerate Mode가 관찰되며 아쉬운 모습을 보이기도 했다.

---

**Normalizing Flows**

- RealNVP: Density estimation using Real NVP, Dinh et al., 2016. [[arXiv:1605.08803](https://arxiv.org/abs/1605.08803)]

VAE가 연구되는 동시에 approximate posterior 도입 없이 marginal $\log p_{\theta,X}(x)$를 구하려는 시도가 있었다.

만약 parametrized generator $G_\theta: Z \to X$가 가역함수(혹은 전단사함수, Bijective)이면 marginal pdf는 변수 치환 법칙에 따라 $p_{\theta,X}(x) = p_Z(f^{-1}(x))\left|\frac{\partial f^{-1}(x)}{\partial x}\right|$를 만족한다.

적분 없이도 determinant of jacobian을 구함으로 marginal을 구할 수 있게 되었고, 이 과정이 differentiable 하다면 gradient 기반의 학습도 가능하다. 문제는 뉴럴 네트워크 가정에서 jacobian을 구한 후, 이미지 pixel-dimension에서 $O(n^3)$의 determinant 연산을 수행해야 한다는 것이다(e.g. 256x256 이미지의 경우 281조, 281 Trillion).

RealNVP는 현실적인 시간 내에 이를 수행하기 위해 Coupling layer를 제안한다.

$$\begin{align*}
y_{1:d} &= x_{1:d} \\\\
y_{d+1:D} &= x_{d+1:D} \odot \exp(s_\theta(x_{1:d})) + t_\theta(x_{1:d})
\end{align*}$$

Affine coupling layer는 hidden state를 반으로 나눠 한 쪽을 유지한 채, 나머지 반에 다른 반을 기반으로 affine transform을 가한다. 이는 가역 연산으로, 절반의 원본을 통해 다른 절반의 역연산이 가능하며, 연산 복잡도 역시 순연산과 동일하다.

$$\begin{align*}
x'_{1:d} &= y _{1:d} \\\\
x' _{d+1:D} &= (y _{d+1:D} - t _\theta(y _{1:d})) \odot \exp(-s _\theta(y _{1:d}))
\end{align*}$$

Affine coupling layer의 Jacobian matrix는 $y_{1:d}$와 $x_{1:d}$가 identity mapping이기에 identity block matrix를 형성, $y_{1:d}$는 $x_{d+1:D}$에 dependent 하지 않기 때문에 zeroing out 되고, $y_{d+1:D}$와 $x_{d+1:D}$는 element-wise linear 관계로 diagonal block matrix가 되어, 최종 low triangular matrix의 형태로 구성된다. 이 경우 determinant는 별도의 matrix transform을 거치지 않고 대각 원소의 곱으로 곧장 연산해 낼 수 있다.

$$\begin{align*}
\frac{\partial y}{\partial x} &= \left[\begin{matrix}
\mathbb I_d & 0 \\\\
\frac{\partial y_{d+1:D}}{\partial x_{1:d}} & \mathrm{diag}[\exp(s(x_{1:d}))]
\end{matrix}\right] \\\\
\det\frac{\partial y}{\partial x} &= \prod_{i=d+1}^D \exp(s_i(x_{1:d})) = \exp\left(\sum^D_{i=d+1}s_i(x_{1:d})\right)
\end{align*}
$$

Affine coupling layer를 여러 개 쌓아 $f_2(f_1(z))$의 형태로 표현한다면, 역함수는 $f_1^{-1}(f_2^{-1}(x))$로 네트워크를 출력부부터 역순으로 연산해 나가면 되고, determinant 역시 각각 계산하여 구할 수 있다.

$$\det\frac{\partial f_2}{\partial z} = \det\frac{\partial f_2}{\partial f_1}\frac{\partial f_1}{\partial z} = \left(\det \frac{\partial f_2}{\partial f_1}\right)\left(\det\frac{\partial f_1}{\partial z}\right)$$

다만 이 경우 한쪽에만 연산이 가해지는 형태이기에, Coupling layer 이후 shuffling $[y_{1:d}, y_{d+1:D}] = [x_{d+1:D}, x_{1:d}]$를 수행하여 각각의 청크가 모두 transform 될 수 있도록 구성한다.

$$\begin{align*}
\max_\theta \log p_{\theta, X}(x) &= \max_\theta \left[\log p_Z(f_\theta^{-1}(x)) + \log\left|\det\frac{\partial f_\theta^{-1}(x)}{\partial x}\right|\right] \\\\
&= \max_\theta \left[\log p_Z(f_\theta^{-1}(z)) - \exp\left(\sum^L_{l=1}\sum^D_{i=d+1}s^l_{\theta,i}(x^l_{1:d})\right)\right]
\end{align*}$$

L개 affine coupling layer w/shuffling으로 구성된 네트워크 $f_\theta$의 최종 objective는 위와 같다. 

Normalzing Flow는 Network의 형태를 제약함으로 Generation과 함께 exact likelihood를 구할 수 있게 되었고, 별도의 Encoder 없이 posterior를 구할 수 있다는 장점이 있다.

하지만 반대로, 네트워크의 형태에 제약을 가하기에 발생하는 approximation의 한계가 발생할 수 있다. 자세한 내용은 뒤에서 논의한다.

---

- Glow: Generative Flow and Invertible 1x1 Convolutions, Kingma & Dhariwal, 2018. [[arXiv:1807.03039](https://arxiv.org/abs/1807.03039)]

Glow는 이에서 더 나아가, 256x256 크기의 이미지까지 연구를 확장하여 그 실용성을 보였다.

기본적으로 가역 함수와 치환 법칙을 기저로 하며, RealNVP에서 몇 가지 네트워크 구조를 수정하였다.

가장 먼저 Batch Normalization을 Activation Normalization으로 교체한다. 당시 GPU VRAM은 10GB (1080TI, 2080TI) 정도로, 이미지의 크기가 조금만 커져도 배치의 크기를 1~2까지로 줄여나가야 했다. 이러한 상황에서 BN의 Moving statistics는 noisy 했고, 성능 하락을 감안해야 했다.

이에 Glow는 최초 Forward pass에서 normalization 직전 레이어의 평균과 표준편차를 연산하여 저장해두고, 이를 토대로 normalization을 수행한다. 한 번 초기화된 파라미터는 이후 별도의 이동 평균 처리나 통계치 재연산을 수행하지 않고, 일반적인 trainable constant로 여긴다. 이를 data-dependent initalization이라 하고, 위 정규화 레이어를 activation normalization이라 한다.

```py {style=github}
# PSEUDO CODE OF DATA-DEPENDENT INITIALIZATION
def __init__(self):
    super().__init__()
    self.mean, self.logstd = None, None

def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
    # x: [B, C, H, W]
    if self.mean is None:
        with no_grad():
            self.register_parameter(
                "mean",
                nn.Parameter(x.mean(dim=[0, 2, 3], keepdim=True)),
            )
            self.register_parameter(
                "logstd",
                nn.Parameter(x.std(dim=[0, 2, 3], keepdim=True).log()),
            )
    norm = (x - self.mean) * (-self.logstd).exp()
    logdet = -self.logstd
    return norm, logdet

def inverse(self, y: Tensor) -> Tensor:
    assert self.mean is not None
    return y * self.logstd.exp() + self.mean
```

ActNorm은 첫 배치에서 zero-mean, unit-variance의 feature map을 반환하여 학습을 안정화하고, 이후는 학습에 따라 자연스레 값을 바꿔나간다. 

FYI. DDI는 Weight normalization[[Salimans & Kingma, 2016.](https://arxiv.org/abs/1602.07868)]에서 효과가 확인된 바 있다.

다음은 Invertible 1x1 convolution이다. RealNVP가 Shuffling을 통해 절반의 feature map에 연산이 가해지지 않던 문제를 해결했다면, Glow는 가역 행렬을 channel-axis에 곱함으로(1x1 conv), 채널 축의 정보 공유를 학습 가능하도록 두었다(generalized permutation).

우선 초기화 단계에서 QR 분해를 통해 1x1 Convolution의 Random weight matrix W가 invertible 하게 두었고, 이후에는 $\log|\det W|$를 직접 연산하여 objective에 활용(`torch.linalg.slogdet`), inference에는 weight의 역행렬을 구하여 활용한다(`torch.linalg.inv`).

다만 이 경우 channel-axis의 크기가 커질 경우 연산량에 부담이 생길 수 있으므로, 다음과 같이 LU Decomposition을 활용하여 연산량을 줄여볼 수도 있다.

```py {style=github}
def __init__(self, channels: int):
    super().__init__()
    weight, _ = torch.linalg.qr(torch.randn(channels, channels))
    p, l, u = torch.linalg.lu(weight)
    self.p = nn.Parameter(p)
    self.l = nn.Parameter(l)
    self.u = nn.Parameter(u)
    self.s = nn.Parameter(torch.diagonal(u))
    self.register_buffer("i", torch.eye(channels), persistent=False)

@property
def weight(self):
    return self.p @ (self.l.tril(-1) + self.i) @ (self.u.triu(1) + self.s)

def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, c, h, w = x.shape
    shuffled = F.conv2d(x, self.weight[..., None, None])
    logdet = self.s.abs().log().sum() * h * w
    return shuffled, logdet
  
def inverse(self, y: torch.Tensor) -> torch.Tensor:
    return F.conv2d(x, torch.linalg.inv(self.weight)[..., None, None])
```

마지막으로 affine coupling layer 두 개 네트워크 $t_\theta, s_\theta$의 마지막 convolution 레이어를 zero-initialize하여 학습의 첫 forward pass에서는 identity mapping이 되도록 구성하였다. 이는 LayerScale[[Touvron et al., 2021.](https://arxiv.org/abs/2103.17239)]처럼 레이어가 많은 네트워크를 운용할 때 학습을 안정화한다고 알려져 있다.

이러한 트릭을 활용하여 Glow는 256x256 이미지에서도 좋은 합성 결과를 보였고, 아직도 likelihood 기반의 새로운 학습 방법론이 소개될 때마다 베이스라인으로 인용되고 있다.

---

- CIF: Relaxing Bijectivity Constraints with Continuously Indexed Normalising Flows, Cornish et al., 2019. [[arXiv:1909.13833](https://arxiv.org/pdf/1909.13833)]

앞서 이야기한 Bijective의 제약에 의해 발생하는 Approximation의 한계에 관하여 이야기해 보고자 한다.

***Exact Support Matching***

Normalizing flows는 대표적인 pushforward measure이다.

i.e. measurable space $(Z, \Sigma_Z, \mu_Z)$, $(X, \Sigma_X)$와 measurable mapping $f: Z \to X$에 대해 $f_\\# p_Z = p_Z(f^{-1}(B)); B \in \Sigma_X$를 Pushforward measure라 한다. (w/sigma algebra $\Sigma_Z, \Sigma_X$ of $Z, X$)

Normalizing flows는 특히 generator $f$를 전단사함수로 가정하기 때문에, $\mathrm{supp}\ p_X$와 $\overline{f(\mathrm{supp}\ p_Z)}$가 같아야 한다. i.e. support of $p_X$, $\mathrm{supp}\ p_X = \\{x \in X : \forall \mathrm{open}\ U \ni x,\ p_X(U) > 0 \\}$, closure of $S$, $\overline S$

FYI. 직관적으로 support는 사건의 발생 확률이 0보다 큰 원소의 집합이다. 확률이 존재하는 공간을 전단사 함수로 대응하였을 때, 대응된 원소 역시 발생 가능성이 0보다 커야 함을 의미한다.

RealNVP, Glow 등의 Normlizing flows는 대부분 연속 함수이다(tanh, relu, sigmoid 등의 연속 활성함수를 사용하는 네트워크 기반의 affine coupling을 가정). 동시에 전단사 함수이기 때문에 역함수 역시 연속 함수이고, 이 경우 $f$는 topological property를 보존하는 homeomorphism이다(위상 동형 사상).

위상 동형인 $Z$와 $X$는 hole의 수, connected component의 수 등이 같아야 한다. 흔히 가정하는 정규 분포의 support는 hole을 가지지 않고, 1개의 connected components를 가진다. 만약 데이터의 분포가 두 Truncate Normal 분포의 mixture로 표현되어 그의 support가 2개의 connected components를 가진다면, 두 분포를 대응하는 연속 함수 형태의 normalizing flows를 construction 하는 데에 한계가 발생한다.

***Lipschitz Constraints***

경우에 따라 Invertible ResNet[[Behrmann et al., 2018.](https://arxiv.org/abs/1811.00995)], Residual Flow[[Chen et al., 2019.](https://arxiv.org/abs/1906.02735)]는 invertibility를 위해 network의 lipschitz constant를 제약한다.

$$\mathrm{Lip}(f) = \sup_{x\ne y}\frac{|f(x) - f(y)|}{|x - y|} \implies |f(x) - f(y)| \le \mathrm{Lip}(f)|x - y|\ \forall x, y$$

Injective $f$에 대해 bi-Lipschitz constant $\mathrm{BiLip}\ f = \max\left(\sup_{z\in Z}|J_{f(z)}|, \sup_{x\in f(Z)}|J_{f^{-1}(x)}|\right)$를 정의하자(w/norm of jacobian $|J_\cdot|$). homeomorphic하지 않은 두 topological space $Z$와 $X$는 $\lim_{n\to\infty}\mathrm{BiLip}\ f_n = \infty$일 때에만 $f_{n}\\#p_Z \stackrel{D}{\to}p_X$의 weak convergence를 보장한다(under statistical divergence $D$, i.e. $D(f_n\\#p_Z, p_X)\to 0$ as $n\to\infty$).

Residual Flow의 각 레이어가 Lipschitz Constant $K$를 갖는다면, N개 레이어로 구성된 네트워크 전체의 Lipschitz Constant는 최대 $K^N$이다. Homeomorphic하지 않은 임의의 pushforward measure $f_n\\#p_Z$를 $p_X$로 근사하기 위해서는 $K^N\to\infty$의 조건이 만족해야 하고, 그에 따라 무수히 많은 레이어를 요구할 수도 있다. 

---

Normalizing flows는 네트워크의 제약상 고질적으로 Exact support matching 문제와 Lipschitz-constrained network의 표현력 문제를 겪게 된다. Continuously Indexed Flow, CIF는 이를 해결하고자 augmentation을 제안한다.

CIF는 bijective generator $G_\theta$를 indexed family $\\{F_{\theta, u}(\cdot): Z \to X\\}_{u\in\mathrm{supp}\ U}$ 로 확장한다. $x = F _\theta(z, u)$의 2 변수 함수를 생각한다면, u가 고정되어 있을 때, $z = F _\theta^{-1}(x; u)$의 가역함수를 고려할 수 있다. $z$와 $x$는 여전히 change-of-variables의 관계인 반면, $u\sim U$는 데이터의 차원과 무관한 잠재 변수이기에 variational inference의 대상이 된다.

prior $p_U(u)$와 approximate posterior $q_\phi(u|x)$를 가정하자. 우리의 목표는 $p_{\theta, X}(x)$이고, graphical model을 가정할 때 joint는 $p_{\theta, X,U}(x, u) = p_U(u)p_Z(F^{-1}_\theta(x; u))\left|\frac{\partial F^{-1} _\theta(x)}{\partial x}\right|$이다. variational lowerbound는 다음과 같이 정리된다.

$$\mathrm E _{u\sim q _\phi(u|x)}\log\frac{p _{\theta, X,U}(x, u)}{q _\phi(u|x)} = \mathbb E _{u\sim q _\phi(u|x)}\left[\log p _U(u) + \log\frac{p_Z(F^{-1} _\theta(x; u))}{q _\phi(u|x)}\right] + \log\left|\det\frac{\partial F^{-1} _\theta(x)}{\partial x}\right|$$

CIF는 여기서 더 발전시켜, prior $p_U(u)$를 $z$ 조건부 분포 $p_{\theta, U|Z}(u|z) = \mathcal N(\mu_\theta(z), \Sigma_\theta(z))$로 두어 학습의 대상으로 삼는다(i.e. $p_{\theta, X, U}(F_\theta(z; u), u) = p_{\theta, U|Z}(u|z)p_Z(z)|J_{F^{-1}_\theta}|$).

이를 통해 저자는 homeomorphic하지 않은 두 공간에 대해서도 $F_\theta$가 $Z$에서 $U$에 대해 항상 Surjective이면, CIF가 Exact support matching을 가능케 함을 보인다. lipschitz constraints가 존재하는 네트워크에서도 동일하다.

$$\forall z\in\mathrm{supp}\ Z,\ F_\theta(z; \cdot): U\to X\ \mathrm{is\ surjective} \iff p_{\theta, X} = p_X$$

Surjective라는 어렵지 않은 조건 내에서 exact match가 가능한 해가 존재함을 보였고, 이의 학습 가능성은 emperical하게 NLL이 낮아짐을 통해 보인다.

이렇게 Normalizing flows에서 데이터의 차원을 넘어 새로운 latent variable을 추가하는 형태를 agumentation이라 하고, ANF[[Huang et al., 2020.](https://arxiv.org/abs/2002.07101)], VFLow[[Chen et al., 2020.](https://arxiv.org/abs/2002.09741)]의 Concurrent work가 존재한다.

이 두 논문 모두 augmented normalizing flow에 관한 emperical study를 보이며, VFlow의 경우 CIF와 유사히 augmented normalizing flow가 vanilla보다 NLL이 더 낮을 수 있음을 보인다.

---

**Continuous Normalizing Flows**

앞서 확인하였듯, Normalizing Flows는 네트워크의 Lipschitz Constant가 제약되었을 때 임의 분포의 근사를 위해 레이어의 수를 무한히 요구할 수 있다. 그렇다면 레이어의 수를 무한히 표현하기 위해서는 어떻게 해야 할까.

두 집합 $Z, X\subset\mathbb R^D$을 연관 짓는 함수 $F_\theta: Z\to X$가 있다 가정하자. 함수 $F_\theta$는 N-Layer Residual Network로, $z\in Z$를 입력으로 값을 순차적으로 변환시켜 끝에 $X$에 도달하게 한다. 이에 $F_\theta$의 i번째 레이어 $f^{(i)}_ \theta = f_ \theta(\cdot; i)$를 $f_\theta(\cdot; t_i)$로 표현하면, $x_{t_{i+1}} = x_{t_i} + f_\theta(x_{t_i}; t_i)$이고, $f_\theta: \mathbb R^D\times I\to\mathbb R^D$이다.

where $t_i = i / N,\ i = 0,1,...,N\implies I = \\{t_i\\}_{i=0}^{N-1};\ x_0 = z,\ x_1\in X$

$Nf_\theta(x_{t_i}; t_i) = N(x_{t_{i+1}} - x_{t_i})$에 대해 $N\to\infty$를 상정하면, 극한이 존재할 때 다음과 같이 모델링할 수 있다.

$$\lim_{\Delta t := N^{-1}\to 0}\frac{x_{t + \Delta t} - x_{t}}{\Delta t} = \frac{dx_t}{dt} = f_\theta(x_t; t): \mathbb R^D\times[0, 1]\to\mathbb R^D \\\\
x_1 - x_0 = \int^1_0\frac{dx_t}{dt}dt = \int^1_0 f_\theta(x_t; t)dt\implies x_1 = x_0 + \int^1_0f_\theta(x_t; t)dt$$

$N\to\infty$ 이므로 레이어 $f^{(i)}_ \theta$의 첨자에 해당하던 $I$는 $[0, 1)$의 구간으로 표현되고, 이제는 N번째 hidden layer가 아닌 어떤 순간 $t$의 $x_t$ 변량 $dx_t/dt$를 $f_ \theta(x_t; t)$의 time-conditional neural network로 표현한다.

각 레이어가 서로 독립된 Subnetwork로 구성되던 기존과 달리, 이제는 모든 시점에서 하나의 네트워크를 공유하고, 대신 현시점 $t\in[0, 1]$를 조건으로 주는 방식이다.

$f_\theta$가 특수한 형태로 제약되지 않는 이상 적분을 통한 mapping은 불가능하므로, Euler solver, Runge-Kutta Method 등 Numerical solver를 통해 approximate 한다. 이 경우 사전에 Discretized points $\\{t_i\\}_{i=1}^N$를 정의해 두었다가, 다음과 같이 1st-order approximate 하는 예시를 들 수 있다.

$$x_{t_{i+1}} = x_{t_i} + (t_{i+1} - t_i)f_\theta(x_{t_i}; t_i);\ \ x_{t_0} = z,\ x_{t_N} = x$$

문제는 어떻게 $f_\theta(x_t; t)$를 학습할 것인지이다.

가장 직관적으로는 Numerical solver를 통해 $\hat x = \text{ODESolver}(f_\theta, z; \\{t_i\\}_{i=1}^N)$를 획득하여 Objective를 취하고, RNN과 같이 BPTT(Back-propagation Through Time)을 통한 업데이트를 고려할 수 있다. 하지만, 이 경우 이미 알려진 문제인 Vanishing, Exploding gradients 등 여러 문제에 직면한다.

- NODE: Neural Ordinary Differential Equations, Chen et al., 2018. [[arXiv:1806.07366](https://arxiv.org/abs/1806.07366)]

이에 대한 Practical solution을 제안한다.

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
- CIF: Relaxing Bijectivity Constraints with Continuously Indexed Normalising Flows, Cornish et al., 2019. [[arXiv:1909.13833](https://arxiv.org/pdf/1909.13833)]
- FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models, Grathwohl et al., 2018. [[arXiv:1810.01367](https://arxiv.org/abs/1810.01367)]
- Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks, Salimans & Kingma, 2016. [[arXiv:1602.07868](https://arxiv.org/abs/1602.07868)]
- LayerScale: Going deeper with Image Transformers, Touvron et al., 2021. [[arXiv:2103.17239](https://arxiv.org/abs/2103.17239)]
- Invertible Residual Networks, Behrmann et al., 2018. [[arXiv:1811.00995](https://arxiv.org/abs/1811.00995)]
- Residual Flows for Invertible Generative Modeling, Chen et al., 2019. [[arXiv:1906.02735](https://arxiv.org/abs/1906.02735)]
- Augmented Normalizing Flows: Bridging the Gap Between Generative Flows and Latent Variable Models, Huang et al., 2020. [[arXiv:2002.07101](https://arxiv.org/abs/2002.07101)]
- VFlow: More Expressive Generative Flows with Variational Data Augmentation, Chen et al., 2020. [[arXiv:2002.09741](https://arxiv.org/abs/2002.09741)]

---

{{< details summary="TODO" >}}

0. Preliminaries

Oksendal SDE
- Brownian Motion Model
- Ito process
- Ito Diffusion, Markovian Property

1. Score model
- Sliced Score Matching: A Scalable Approach to Density and Score Estimation, Song et al., https://arxiv.org/abs/1905.07088
- Generative Modeling by Estimating Gradients of the Data Distribution, Song & Ermon, https://arxiv.org/abs/1907.05600

2. DDPM
- Denoising Diffusion Probabilistic Models, Ho et al., 2020. https://arxiv.org/abs/2006.11239, https://revsic.github.io/blog/diffusion/
- Diffusion Models Beat GANs on Image Synthesis, Dhariwal & Nichol, 2021. https://arxiv.org/abs/2105.05233
- Variational Diffusion Models, Kingma et al., 2021. https://arxiv.org/abs/2107.00630, https://revsic.github.io/blog/vdm/
- Denoising Diffusion Implicit Models, Song et al., 2020. https://arxiv.org/abs/2010.02502
- Classifier-Free Diffusion Guidance, Ho & Salimans, 2022. https://arxiv.org/abs/2207.12598
- EDM: Elucidating the Design Space of Diffusion-Based Generative Models, Karras et al., 2022. https://arxiv.org/abs/2206.00364
- EDM2: Analyzing and Improving the Training Dynamics of Diffusion Models, Karras et al., 2023. https://arxiv.org/abs/2312.02696
- [Blog] Essay: VAE as a 1-step Diffusion Model
, https://revsic.github.io/blog/1-step-diffusion/

3. SDE & PF ODE
- Score-Based Generative Modeling through Stochastic Differential Equations, Song et al., 2020. https://arxiv.org/abs/2011.13456

4. Rectified Flow & Flow Matching
- Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow, Liu et al., 2022. https://arxiv.org/abs/2209.03003
- Flow Matching for Generative Modeling, Lipman et al., 2022. https://arxiv.org/abs/2210.02747
- Simple ReFlow: Improved Techniques for Fast Flow Models, Kim et al., 2024. https://arxiv.org/abs/2410.07815s
- Improving the Training of Rectified Flows, Lee et al., 2024. https://arxiv.org/abs/2405.20320
- CAF: Constant Acceleration Flow, Park et al., https://arxiv.org/abs/2411.00322

5. Consistency Models
- Consistency Models, Song et al., 2023. https://arxiv.org/abs/2303.01469, https://revsic.github.io/blog/cm/
- Inconsistencies In Consistency Models: Better ODE Solving Does Not Imply Better Samples, Vouitsis et al., 2024. https://arxiv.org/abs/2411.08954
- ECT: Consistency Models Made Easy, Geng et al., 2024. https://arxiv.org/abs/2406.14548
- Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models, Lu & Song, 2024. https://arxiv.org/abs/2410.11081
- Improving Consistency Models with Generator-Augmented Flows, Issenhuth et al., https://arxiv.org/abs/2406.09570

6. Bridge
- Diffusion Schrodinger Bridge Matching, Shi et al., 2023. https://arxiv.org/abs/2303.16852
- Consistency Diffusion Bridge Models, He et al., 2024. https://arxiv.org/abs/2410.22637

7. Furthers
Unified view
- SurVAE Flows: Surjections to Bridge the Gap between VAEs and Flows, Nielsen et al., 2020. https://arxiv.org/abs/2007.02731, https://revsic.github.io/blog/survaeflow/
- Simulation-Free Training of Neural ODEs on Paired Data, Kim et al., 2024. https://arxiv.org/abs/2410.22918
- Simulation-Free Differential Dynamics through Neural Conservation Laws, Hua et al., ICLR 2025. https://openreview.net/forum?id=jIOBhZO1ax
- Adversarial Likelihood Estimation With One-Way Flows, Ben-Dov et al., 2023. https://arxiv.org/abs/2307.09882

Fewer-step approaches
- Progressive Distillation for Fast Sampling of Diffusion Models, Salimans & Ho, 2022. https://arxiv.org/abs/2202.00512
- Tackling the Generative Learning Trilemma with Denoising Diffusion GANs, Xiao et al., 2021.
- InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation, Liu et al., 2023. https://arxiv.org/abs/2309.06380
- One Step Diffusion via Shortcut Models, Frans et al,. 2024. https://arxiv.org/abs/2410.12557
- One-step Diffusion with Distribution Matching Distillation, Yin et al., 2023. https://arxiv.org/abs/2311.18828
- Improved Distribution Matching Distillation for Fast Image Synthesis, Tianwei Yin et al., 2024. https://arxiv.org/abs/2405.14867
- One-step Diffusion Models with f-Divergence Distribution Matching, Xu et al., 2025. https://arxiv.org/abs/2502.15681

First-order ODE
- Rectified Diffusion: Straightness Is Not Your Need in Rectified Flow, Want et al., 2024. https://arxiv.org/abs/2410.07303
- Consistency Flow Matching: Defining Straight Flows with Velocity Consistency, Yang et al., 2024. https://arxiv.org/abs/2407.02398
- One Step Diffusion via Shortcut Models, Frans et al., https://arxiv.org/abs/2410.12557

Etc
- The GAN is dead; long live the GAN! A Modern GAN Baseline, Huang et al., https://arxiv.org/abs/2501.05441
- IMM: Inductive Moment Matching, Zhou et al., https://arxiv.org/abs/2503.07565

- [Blog] Essay: Generative models, Mode coverage, https://revsic.github.io/blog/coverage/

{{</details>}}
