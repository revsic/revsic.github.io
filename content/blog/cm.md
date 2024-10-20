---
title: "[WIP] Consistency Models"
date: 2024-10-20T21:48:29+09:00
draft: false

# post thumb
image: "images/post/cm/header.png"

# meta description
description: "Consistency Models"

# taxonomies
categories:
  - "Bayesian"
tags:
  - "Machine Learning"
  - "Deep Learning"
  - "Generative"
  - "Bayesian"
  - "Consistency Model"

# post type
type: "post"
---

**아래는 논문을 보며 작성한 초안입입니다. 비문과 맞춤법 오류를 포함하며, 근시일 내에 업데이트할 예정입니다.*

- Consistency Models, Song et al., 2023. [[arXiv:2303.01469](https://arxiv.org/abs/2303.01469)]
- Keyword: Consistency Models, Probability Flow ODE
- Problem: Slow generation of Diffuison Models.
- Solution: TBD
- Benefits: TBD
- Contribution: TBD

**Introduction**

Consistency models(이하 CM)의 목적은 trajectory 위의 모든 점을 trajectory의 시점으로 매핑시키는 것

Trajectory의 종점에 존재하는 tractable noise distribution으로 부터 시점에 해당하는 data distribution sample을 획득하는 것이 목표.

첫 방법론은 Pretrained Diffusion model + ODE Solver을 두고 trajectory 위 데이텀 포인트를 획득해서 Distillation을 시키는 것.

두 번째 방법론은 Unbiased estimator를 통해 Score를 획득하고, Consistency model을 학습하는 것

아쉬운 점은 데이터셋이 대개 작다. (e.g. cifar-10 32x32, imagenet 64x64, lsun 256x256)

fyi. Iterative sampling이 오히려 inverse problem과 editing을 가능케 했다고도 보는 것 같다.

**Diffusion Models**

CM은 continuous diffusion에서 영감을 많이 받았다. 다음은 Data distribution에 취할 Diffusing SDE

$$\mathrm{d}x_t = \mu(x_t, t)\mathrm{d}t + \sigma(t)\mathrm{d}w_t\ \mathrm{where}\ t\in [0, T],\ T > 0$$

이때 $w_t$는 standard brownian motion. Karras at al., 2022에서는 drift $\mu(x, t) = 0$, diffusion $\sigma(t) = \sqrt{2t}$로 가정하여 $p_t(x) = p_\mathrm{data}(x) \otimes \mathcal N(0, t^2I)$의 variance exploding diffusion을 가정. (i.e. tractable prior를 $\mathcal N(0, T^2I)$로 가정)

위 SDE는 Reverse process에 대한 ODE의 존재성을 보장. (이하 Probability Flow ODE)

$$\mathrm{d}x_t = \left[\mu(x_t, t) - \frac{1}{2}\sigma(t)^2\nabla\log p_t(x_t)\right]$$

Diffusion Models 혹은 Score Models는 $\nabla\log p_t(x_t)$를 모델링 하고자 함, i.e. Score, $s_\phi(x, t)$. Drift, diffusion term을 직접 대입하면 $\frac{\mathrm{d}x_t}{\mathrm{d}t} = -ts_\phi(x_t, t)$를 모델링하는 것과 동치(이하 Emperical PF ODE).

$\hat x_T \sim \pi = \mathcal N(0, T^2I)$로 ODE를 초기화하면, numerical ode solver에 통해 데이텀 포인트를 획득 가능(e.g. Euler solver, Heun solve 등. 실제 테스트에서는 Heun + N=18일 떄가 best choice).

Numerical stability를 위해 $t = \epsilon$에서 정지 ($\because \hat x_0 \sim \mathcal N(0, 0)$) \
(image를 [-1, 1]의 pixel value로 rescale, T=80, $\epsilon$=0.002로 가정)

ODE Solver 활용시 10회 이상 샘플링이 필요로 하는 상황. 더 step을 줄이기 위해 distillation을 수행할 경우 large dataset을 요구하기도 한다. progressive distillation가 CM 당시까지는 가장 현실적 대안

**Consistency Models**

$\\{x_t\\}_ {t\in [\epsilon, T]}$의 Solution trajectory에 대해 Consistency function $f: (x_t, t)\mapsto x_\epsilon$를 정의

이는 same trajectory의 두 시점 $t, t'\in[\epsilon, T]$에 대해 $f(x_t, t) = f(x_{t'}, t')$의 consistency를 보장.

{{< figure src="/images/post/cm/1.png" width="80%" caption="Generation trajectory (Figure 2, Song et al., 2023)" >}}

Neural ODE 같은 관련 연구와 다른 점은 invertibility를 가정하지 않은 것

$t\in[\epsilon, T]$의 범위를 가정할 때, $f(x_\epsilon, \epsilon) = x_\epsilon$으로 identity function을 가정(boundary condition). Free-form NN $F_\theta(x, t)$에 대해 다음으로 표현 가능 

$$f_\theta(x, t) = \left\\{\begin{matrix}x & t=\epsilon \\\\ F_\theta(x, t) & t\in(\epsilon, T]\end{matrix}\right.$$

혹은 다음과 같이도 표현

$$f_\theta(x, t) = c_\mathrm{skip}(t)x + c_\mathrm{out}(t)F_\theta(x, t)$$

이때 $c_\mathrm{skip}(t)$와 $c_\mathrm{out}(t)$는 $c_\mathrm{skip}(\epsilon) = 1,\ c_\mathrm{out}(\epsilon) = 0$이고, 미분 가능한 함수. 디퓨전 모델에서 이러한 Formulation을 많이 활용해 왔기에, CM은 후자를 사용

Sampling: $\hat x_\epsilon = f_\theta(\hat x_T, T)\ \mathrm{where}\ \hat x_T \sim \mathcal N(0, T^2I)$로 single step에도 획득 가능.

multistep으로 운용할 경우 $\hat x_{\tau_n} \leftarrow x + \sqrt{\tau^2_n - \epsilon^2}z,\ x \leftarrow f_\theta(\hat x_{\tau_n}, \tau_n)$

$\\{\tau_1, ..., \tau_{N - 1}\\}$로 N스텝 샘플링을 수행하고, 이 수열은 FID를 최적화하는 수열로 emprical하게 찾음

- Q: 왜 Multistep을 가정하는지, 왜 성능상 이점을 가지게 되는지

Prior distribution(혹은 latent space) 내에서 interpolation, inpainting, colorization, super-resolution, stroke-guided image editing 등이 가능.

fyi. 대개 score model에서 classifier-guidance 하던 것들은 모두 가능할 것

**Training**

$$t_i = (\epsilon^{1/\rho} + \frac{i-1}{N-1}(T^{1/\rho} - \epsilon^{1/\rho}))^\rho,\ \mathrm{where}\ \rho=7$$

pretrained score model $s_\phi(x, t)$가 있을 때 discretizing horizon $[\epsilon, T]$ into $N - 1$을 수행.

ODE solver의 update function을 Euler Solver로 가정 시 (i.e. $\Phi(x, t; \phi) = -ts_\phi(x, t)$), Emperical PF ODE의 Trajectory 위 인접 샘플 포인트 $\hat x^\phi_{t_n} = x_{t_{n+1}} - (t_n - t_{n+1})t_{n+1}s_\phi(x_{t_{n+1}}, t_{n+1})$를 획득 가능

$x$는 데이터에서 샘플링, $x_{t_{n+1}}$은 transition density $\mathcal N(x, t^2_{n+1}I)$에서 샘플링한 후, $(\hat x^\phi_{t_n}, x_{t_{n+1}})$의 adjacent data point pairs를 확보. 

$$\mathcal L^N_{CD}(\theta, \theta^-; \phi) := \mathbb E_{x\sim p_{data}, n\sim \mathcal U(1, N-1)}[\lambda(t_n)d(f_\theta(x_{t_{n+1}}, t_{n+1}), f_{\theta^-}(\hat x^\phi_{t_n}, t_n))]$$

이때 $\lambda(\cdot)\in\mathbb R^+$, $\theta^-$는 $\theta$의 running average, $d(\cdot, \cdot)$은 metric function.

Metric으로는 L2, L1, LPIPS를 후보로, $\lambda(\cdot) = 1$, $\theta^-$는 EMA로 가정, i.e. $\theta^- \leftarrow \mathrm{stopgrad}(\mu\theta^- + (1 - \mu)\theta)$

fyi. Metric은 LPIPS에서 가장 높은 성능을 보임

Analysis를 통해 $f_\theta$가 lipschitz constant를 가질 때 다음을 만족함을 확인
$$\mathcal L^N_{CD}(\theta, \theta; \phi) = 0 \rightarrow \sup_{n, x}||f_\theta(x, t_n) - f(x, t_n; \phi)||^2 = O((\Delta t)^p)\ \mathrm{with}\ p \ge 1$$

$\theta^-$가 EMA이므로, 수렴 상황에서 $\theta = \theta^-$를 가정할 수 있고, T가 충분히 클 때 CM의 성능이 arbitarily accurate 해질 수 있음을 방증. 또한 $f_\theta(x, \epsilon) = x$의 identity boundary condition으로 인해 $f_\theta(x, t) = 0$이 되는 trivial solution은 고려하지 않아도 된다.

경우에 따라 극한을 취해 $N\to\infty$ Continuous-time CM을 가정할 수 있지만, jacobian vector product에 대한 미분이 필요하여 현대 딥러닝 프레임워크에 한계가 있고, 이는 appendix에서 효율적 구현을 논의

Distillation을 하지 않을 경우, unbiased score estimation을 통해 운영

$$\nabla\log p_t(x_t) = -\mathbb E_{x\sim p_{data}, x_t\sim \mathcal (x; t^2I)}\left[\frac{x_t - x}{t^2}|x_t\right]$$

이는 Euler solver w/$N \to \infty$에서 pretrained score model을 대체할 수 있음

- Q. 왜 극한을 가정해야 하는지

대신 이 경우 $f_{\theta^-}$가 twice continuously differentiable with bounded second derivatives여야 하고, bounded $\lambda(\cdot),\ \mathbb E[||\nabla\log p_{t_n}(x_{t_n})]||^2_2$와 $\forall t\in[\epsilon, T]: s_\phi(x, t) = \nabla\log p_t(x)$에 한하여 다음을 만족

$$\mathcal L^N_{CD}(\theta, \theta^-; \phi) = \mathcal L^N_{CT}(\theta, \theta^-) + o(\Delta t)$$

또한 $\inf_N\mathcal L^N_{CD}(\theta, \theta^-; \phi) > 0 \to \mathcal L^N_{CT}(\theta, \theta^-)\ge O(\Delta t)$. 이는 distillation에 비해 느리게 학습될 것임을 방증.

직관 상 N이 작을 때에는 training loss가 variance보단 bias가 클 것이기에, 학습 초반에 Distillation에 비해 빠르게 수렴할 것. 반대로 N이 크면 variance가 더 커질 것이라는 직관.

그렇기에 N은 점차 늘리는 방향으로 scheduling하고, 그에 따라 $\mu$도 조정되어야 할 것(N에 반비례하여 감소)

$$N(k) = \left\lceil\sqrt{\frac{k}{K}((s_1 + 1)^2 - s^2_0) + s_0^2} - 1\right\rceil + 1 \\\\
\mu(k) = \exp\left(\frac{s_0\log \mu_0}{N(k)}\right)$$

Continuous-time이 되면 schedule function이 필요하지 않지만, 여전히 efficient implementation이 필요하다.

**Experiments**

{{< figure src="/images/post/cm/2.png" width="100%" caption="Figure3: Various factors that affect CD and cT / Figure4: Multistep generation. (Song et al., 2023)" >}}

Consistency training(이하 CT)은 실제로 N에 굉장히 민감. 작으면 빠르게 수렴하지만 퀄리티가 낮고, 크면 느리게 수렴하는 대신 퀄리티가 상대적으로 높다. 이것이 $N$과 $\mu$를 스케줄링하게 된 계기. 

LPIPS 활용은 Progressive Distillation(이하 PD)에서도 유의미한 개선을 보임. Consistency Distillation(이하 CD)은 PD에 비해 uniform 하게 좋은 성능을 보였고, 샘플링 스텝이 늘어남에 따라 실제로 성능 향상을 보임. 
