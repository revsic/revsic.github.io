---
title: "ANF, VFlow"
date: 2021-03-09T21:25:22+09:00
draft: true

# post thumb
image: "images/post/anfvf/head.jpg"

# meta description
description: "Augmented Normalizing Flow and VFlow."

# taxonomies
categories:
    - "Bayesian"
tags:
    - "Machine Learning"
    - "Deep Learning"
    - "Bayesian"
    - "Normalizing Flow"
    - "Augmented Normalizing Flow"
    - "VFlow"

# post type
type: "post"
---

- ANF, Augmented Normalizing Flows: Bridging the Gap Between Generative Flows and Latent Variable Models, Huang et al., 2020, [arXiv](https://arxiv.org/abs/2002.07101)
- VFlow: More Expressive Generative Flows with Variational Data Augmentation, Chen et al., 2020, [arXiv](https://arxiv.org/abs/2002.09741)
- Keyword: Bayesian, Normalizing Flow, ANF, VFlow
- Problem:
- Solution:
- Benefits:
- Weakness or Future work: -

**Series: Normalizing flow**
1. Normalizing flow, Real NVP [[link](../realnvp)]
2. Glow, Flow++ [[link](../glowflowpp)]
3. ANF, VFlow [this]
4. i-ResNet, ResFlow [future works]
5. CIF, SurVAE Flows [future works]

**Normalizing flow - Bottleneck problem**

Normalizing flow는 latent variable model의 한 축으로 자리 잡아 가고 있다. bijective를 통한 change of variables를 근간으로 하기에, 1) network의 inversion이 efficient 해야 하고, 2) log-determinant of jacobian 연산이 tractable해야 하며, 3) 네트워크가 충분히 expressive 해야 한다.

1)과 2)를 위해서는 기존과 같은 nonlinearity 기반의 레이어를 활용할 수 없었기에, 주로 jacobian의 형태를 제약하는 방식의 부가적인 engineering이 요구되었다. 

이 과정에서 mapping의 형태에 제약이 발생했고, 이에 따른 표현력 절감을 완화하기 위해 Glow[3], Flow++[4]과 같이 engineered bijective에 대한 연구가 등장했다. 

bijective로의 constraint는 tensor의 dimension도 바꿀 수 없게 하였다. 간단하게 tensor slice를 상정한다면, inverse 과정에서 유실된 slice를 복원해야 하고, 충분히 잘 구성된 상황을 가정하지 않은 이상, 이 과정은 analytic하게 구성되지 않을 것이다. 

$$y = x_{1:d} \ \ \mathrm{where} \ x \in \mathbb R^D, \ d < D \\\\
x_{1:d} = y, \ \ x_{d+1:D} = \ ?$$

Universal approximation theorem에서부터 WideResNet[5]으로 이어져 오면서 network의 width는 성능에 중요한 factor가 되었다.

이러한 상황에서 tensor의 dimension을 임의로 조작하지 못해 하위 flow에 internal hidden layers의 higher dimensional feature를 충분히 전달하지 못하면, 매번 부족한 정보를 local-dependency부터 다시 추출해야 할 것이고, 개개 블럭의 표현력이 떨어진 flow는 block의 수를 늘림으로서 이를 해결해야 했다. 그리고 이는 computational inefficiency로 이어진다. 

이 때문에 Flow++[4]에서는 global-dependency를 보다 효율적으로 탐색하기 위해 Transformer[6] 기반의 internal network를 제안하기도 한다.

Dimension problem, bottleneck problem의 요점은 high-resolution, low-dimension의 입력에서부터 high-dimension의 feature를 연산하고, 재사용 가능한지에 존재한다.

Augmented Normalizing Flow, 이하 ANF[1]와 VFlow[2]는 서로 다른 논문이지만 normalizing flow의 dimension 문제에 대해 augmentation이라는 동일한 해결책을 제시한다.

**Augmentation - ANF Perspective**

ANF[1]는 dimension을 늘리기 위해 독립 변수 $e \sim q(e) = \mathcal N(0, I)$를 상정하고, family of joint density models $\\{ p_\pi(x, e): \ \pi \in \mathfrak B\mathcal{(X \times E)} \\}$를 구성한다. 그리고 marginal likelihood 대신에 joint likelihood를 직접 maximize 한다.

$$\hat\pi_\mathcal{A} := {\arg\max}_{\pi \in \mathfrak B(\mathcal{X\times E})}\mathbb E_{(x, e) \sim \hat q(x)q(e)}[\log p_\pi(x, e)]$$

이렇게 확장된 estimator을 ANF[1]에서는 Augmented Maximum Likelihood Estimator (AMLE)라 명명한다.

이 때 model parameter $\pi, \hat\pi_\mathcal{A}$에 대해 상수인 entropy $H(e)$를 활용하여 maximizer로 $\mathcal{L_A}(\pi; x) := \mathbb E_e[\log p_\pi(x, e)] + H(e)$를 상정하면, marginal 과의 차이는 KL divergence로 유도된다.

$$\begin{align*}
&\log p_\pi(x) - \mathcal{L_A}(\pi; x) \\\\
&= \log p_\pi(x) - \mathbb E_e[\log p_\pi(x) + \log p_\pi(e|x)] - H(e) \\\\
&= D_\mathrm{KL}(q(e)||p_\pi(e|x))
\end{align*}$$

이 때의 KL로 유도된 격차를 원문에서는 Augmentation Gap이라 칭한다.

이렇게 되면 exact marginal likelihood를 연산할 수 없으므로, $e_j \sim q(e)$의 K개 i.i.d. sample을 통해 estimate 한다.

$$\hat{\mathcal L_{A, K}} := \log\frac{1}{J}\sum^K_{j=1}\frac{p_\pi(x, e_j)}{q(e_j)}$$

네트워크는 affine coupling으로 구성하며, Glow[3]에서 concat-split을 활용했던 것과 유사히 두개의 block $x$와 $e$를 두고 coupling을 진행한다.

$$\begin{align*}
&g_\pi^\mathrm{enc}(x, e) = \mathrm{concat}(x, s_\pi^\mathrm{enc}(x) \odot e + m_\pi^\mathrm{enc}(x)) \\\\
&g_\pi^\mathrm{dec}(x, e) = \mathrm{concat}(s_\pi^\mathrm{dec}(e)\odot x + m_\pi^\mathrm{dec}(e), e) \end{align*} \\\\
G_\pi = g_{\pi_N}^\mathrm{dec} \circ g_{\pi_N}^\mathrm{enc} \circ ... \circ g_{\pi_1}^\mathrm{dec} \circ g_{\pi_1}^\mathrm{enc}$$

**VFlow Perspective**

VFlow[2] 또한 마찬가지로 additional random variable $z \in \mathbb R^{D_z}$를 상정하고, data $x \in \mathbb R^{D_X}$와 augmented distribution $p(x, z; \theta)$을 구성한다.

$$e = f(x, z; \theta) \in \mathbb R^{D_X + D_Z}$$

이렇게 되면 marginal $\log p(x; \theta) = \log \int p(x, z; \theta)dz$가 intractable 하기에, variational $q(z|x; \phi)$를 상정하고, lower bound를 objective로 구성한다. 

 $$\log p(x; \theta) \ge \mathbb E_{q(x|z; \phi)}[\log p(x, z; \theta) - \log q(z|x; \phi)]$$

마찬가지로 density estimation은 sampling을 통해 진행한다.

$$\log p(x; \theta) \simeq \log\left(\frac{1}{S}\sum^S_{i=1}\frac{p(x, z_i; \theta)}{q(z_i|x; \phi)}\right) \ \ \mathrm{where} \ \ z_1, ..., z_S \sim q(z|x; \phi)$$

이 때 variational $q(z|x; \phi)$는 또 다른 conditional flow로 구성한다.

$$z = g^{-1}(e_q; x, \phi) \Rightarrow \log q(z|x; \phi) = \log p_\epsilon(e_q) - \log\left|\frac{\partial z}{\partial e_q}\right|$$

**Between ANF and VFlow**

두 접근 모두 augmentation을 통해 bottleneck problem을 풀었다는 것에는 동일하나, formulation이 사뭇 다르게 보인다.

ANF의 경우에는 $q(e)$를 standard normal로 가정하여, entropy of e를 통해 lower bound를 산출해 낸다. 이 경우 augmentated gap $D_\mathrm{KL}(q(e)||p_\pi(e|x))$이 $x$에 독립인 marginal $q(e)$를 모델링하는 과정에서의 incapability에 의해 발생하고, 이를 inference suboptimality라 표현한다.

하지만 VFlow의 경우에는 augmented distribution을 variational $q(z|x)$로 상정하여 intractable marginal의 lower bound에 접근하면서 augmented gap $D_\mathrm{KL}(q_\phi(z|x)||p(z|x))$을 줄일 가능성을 제시한다.

이 두 formulation을 보면 언뜻 ANF는 joint를 VFlow는 marginal을 학습하는 차이가 있어 보이지만, entropy가 더해진 ANF의 maximizer $\mathcal{L_A}(\pi; x)$는 사실 variational distribution을 $q(z|x) = p(z) = \mathcal N(0, I)$의 independent standard gaussian으로 상정한 VFlow의 marginal fomulation과 동일하다.

$$\begin{align*}
&\log p(x; \theta) \\\\
&\ge \mathbb E_{z\sim q(z|x)}[\log p(x, z; \theta) - \log q(z|x)]\\\\
&= \mathbb E_{z \sim p(z)}[\log p(x, z; \theta)] + \mathbb E_{z \sim p(z)}[- \log p(z)] \\\\
&= \mathbb E_{z \sim p(z)}[\log p(x, z; \theta)] + H(z) \\\\
&= \mathcal{L_A}(\theta;x)
\end{align*}$$

즉 ANF는 non-parameteric standard gaussian을, VFlow는 conditional flow를 기반으로 한 variational distribution을 상정한 것에 차이가 있다.

**Connection to Vanilla Generative Flows**



**Connection to VAE**

VAE는 1-step flow의 special case로 볼 수 있다. joint distribution을 gaussian factorizing $p(x, z) = \mathcal N(z; 0, I)\mathcal N(x; \mu(z), \exp(\sigma(z))^2)$ 하여 1-step vflow로 구성하면 variational $q(z|x)$에 대해 Gaussian VAE로 볼 수 있다.

$$\epsilon_Z \sim \mathcal N(0, I), \ \ \epsilon_X \sim \mathcal N(0, I) \\\\
z = \epsilon_Z, \ \ x = \mu(\epsilon_Z) + \exp(s(\epsilon_Z)) \circ \epsilon_X$$



**Hierarchical ANF**

**Modeling Discrete Data**

**Experiments**

**Reference**

[1] Huang, C., Dinh, L. and Courville, A. Augmented Normalizing Flows: Bridging the Gap Between Generative Flows and Latent Variable models. 2020. \
[2] Chen, J., et al. VFlow: More Expressive Generative Flows with Variational Data Augmentation. In ICML 2020. \
[3] Kingma, D. P. and Dhariwal, P. Glow: Generative Flow with Invertible 1x1 Convolutions. In NIPS 2018. \
[4] Ho, J. et al. Flow++: Improving flow-based generative models with variational dequantization and architecture design. In ICML 2019. \
[5] Zagoruyko, S. and Komodakis, N. Wide Residual Networks. 2016. \
[6] Vaswani, A., et al. Attention is all you need. In NeurIPS 2017.
