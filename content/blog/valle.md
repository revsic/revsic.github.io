---
title: "Essay: VALL-E, Residual quantization and DDPM"
date: 2023-01-22T16:09:15+09:00
draft: true

# post thumb
image: "images/post/valle/valle.png"

# meta description
description: "VALL-E, Residual quantization and DDPM"

# taxonomies
categories:
  - "Generative"
tags:
  - "Machine Learning"
  - "Deep Learning"
  - "VALL-E"
  - "EnCodec"
  - "Audio Compression"
  - "Residual Quantization"
  - "DDPM"
  - "Diffusion"

# post type
type: "post"
---

아래 글은 비공식적인 개인의 사견임을 밝힌다.

- Essay of residual quantization and DDPM.
- Keyword: VALL-E, EnCodec, Residual Quantization, DDPM, Diffusion

**Introduction**

23년 1월 VALL-E[[arXiv:2301.02111](https://arxiv.org/abs/2301.02111)]라는 One-shot multi-speaker TTS 모델이 나왔다. 

TTS는 기본적으로 텍스트를 입력으로 받아 음성을 합성한다. 텍스트는 20~50hz, 음성은 16k~48khz 정도이기에 텍스트와 음성 사이의 correlation을 학습하는데 어려움이 있다. 이를 해결하기 위해 대부분의 Neural TTS 모델은 중간 매개를 활용한다. 대체로 50~70hz 정도의 spectrogram을 활용하고, text로부터 spectrogram을 생성하는 acoustic model과 spectrogram을 음성 신호로 복원하는 vocoder model의 two-stage 구조를 가정한다.

VALL-E는 중간 매개로 spectrogram 대신 Meta AI의 EnCodec[[arXiv:2210.13438](https://arxiv.org/abs/2210.13438)]을 활용한다. EnCodec은 neural audio compressor로 quantized vector를 bottleneck으로 두는 Auto-encoder 모델이다.

VALL-E는 text와 reference audio를 입력으로 75hz 정도 되는 quantized vector의 index를 추론하도록 학습하고, EnCodec을 통해 quantized vector에서 audio를 복원한다.

이번 글에서는 EnCodec이 어떻게 vector quantization을 수행하는지와 VALL-E가 이를 어떻게 모델링하는지, 해당 방식이 DDPM과 어떤 부분에서 유사한지 논의한다.

**Residual Vector Quantization**

[WIP]

**VALL-E Decoder**

**Denoising Diffusion Models**

**WaveFlow**

**Wrap up**

**Reference**
- VALL-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers, Wang et al., 2023. [[arXiv:2301.02111](https://arxiv.org/abs/2301.02111)]
- EnCodec: High Fidelity Neural Audio Compression, Defossez et al., 2022. [[arXiv:2210.13438](https://arxiv.org/abs/2210.13438)]
- WaveFlow: A Compact Flow-based Model for Raw Audio, Ping et al., 2019. [[arXiv:1912.01219](https://arxiv.org/abs/1912.01219)]


