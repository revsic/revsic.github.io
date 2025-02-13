---
title: "Diffusion Survey"
date: 2025-02-09T13:09:43+09:00
draft: true

# post thumb
# image: "images/post/coverage/trilemma.png"

# meta description
description: "Diffusion Survey"

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

# post type
type: "post"
---

0. Preliminaries

Oksendal SDE
- Brownian Motion Model
- Ito process
- Ito Diffusion, Markovian Property

VAE
- Auto-Encoding Variational Bayes, Kingma & Welling, 2013. https://arxiv.org/abs/1312.6114
- NVAE: A Deep Hierarchical Variational Autoencoder, Vahdat, Kautz et al., 2020. https://arxiv.org/abs/2007.03898

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
- Score-Based Generative Modeling through Stochastic Differential Equations, Song et al., https://arxiv.org/abs/2011.13456

2. DDPM
- Denoising Diffusion Probabilistic Models, Ho et al., 2020. https://arxiv.org/abs/2006.11239, https://revsic.github.io/blog/diffusion/
- Diffusion Models Beat GANs on Image Synthesis, Dhariwal & Nichol, 2021. https://arxiv.org/abs/2105.05233
- Variational Diffusion Models, Kingma et al., 2021. https://arxiv.org/abs/2107.00630, https://revsic.github.io/blog/vdm/
- Denoising Diffusion Implicit Models, Song et al., 2020. https://arxiv.org/abs/2010.02502
- Classifier-Free Diffusion Guidance, Ho & Salimans, 2022. https://arxiv.org/abs/2207.12598
- [Blog] Essay: VAE as a 1-step Diffusion Model
, https://revsic.github.io/blog/1-step-diffusion/

3. PF ODE
- Score-Based Generative Modeling through Stochastic Differential Equations, Song et al., 2020. https://arxiv.org/abs/2011.13456

4. Rectified Flow & Flow Matching
- Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow, Liu et al., 2022. https://arxiv.org/abs/2209.03003
- Flow Matching for Generative Modeling, Lipman et al., 2022. https://arxiv.org/abs/2210.02747

5. Consistency Models
- Consistency Models, Song et al., 2023. https://arxiv.org/abs/2303.01469, https://revsic.github.io/blog/cm/
- Inconsistencies In Consistency Models: Better ODE Solving Does Not Imply Better Samples, Vouitsis et al., 2024. https://arxiv.org/abs/2411.08954

6. Bridge

7. Furthers
Unified view
- SurVAE Flows: Surjections to Bridge the Gap between VAEs and Flows, Nielsen et al., 2020. https://arxiv.org/abs/2007.02731, https://revsic.github.io/blog/survaeflow/

Fewer-step approaches
- Progressive Distillation for Fast Sampling of Diffusion Models, Salimans & Ho, 2022. https://arxiv.org/abs/2202.00512
- Tackling the Generative Learning Trilemma with Denoising Diffusion GANs, Xiao et al., 2021.
- InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation, Liu et al., 2023. https://arxiv.org/abs/2309.06380

Velocity consistency
- Rectified Diffusion: Straightness Is Not Your Need in Rectified Flow, Want et al., 2024. https://arxiv.org/abs/2410.07303
- Consistency Flow Matching: Defining Straight Flows with Velocity Consistency, Yang et al., 2024. https://arxiv.org/abs/2407.02398

- [Blog] Essay: Generative models, Mode coverage, https://revsic.github.io/blog/coverage/
