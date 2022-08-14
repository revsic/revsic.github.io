---
title: "[WIP] Survey: Neural TTS and Attention Alignment"
date: 2022-08-08T23:31:40+09:00
draft: false

# post thumb
image: "images/post/surveytts/timeflow.png"

# meta description
description: "Survey: Neural TTS and Attention Alignment"

# taxonomies
categories:
  - "Attention"
tags:
  - "Machine Learning"
  - "Deep Learning"
  - "TTS"
  - "Attention"
  - "Alignment"
  - "Monotonic"

# post type
type: "post"
---

- Survey of Neural Text-to-Speech models and Attention Alignment
- Keyword: TTS, Attention, Alignment

**Introduction**

16년도 WaveNet[[arXiv:1609.03499](https://arxiv.org/abs/1609.03499)], 17년도 Tacotron[[arXiv:1703.10135](https://arxiv.org/abs/1703.10135)]을 기점으로 딥러닝 기반의 음성 합성 TTS 모델들이 현재까지 꾸준히 발전해 오고 있다. 19년도부터 21년도까지 음성 합성 연구원으로 재직하며 보고 느꼈던 TTS의 발전에 관해 정리해보고자 한다.

**TTS: Text-to-Speech**

TTS는 텍스트를 조건으로 발화 음성을 합성하는 생성 분야를 이야기할 수 있다. 

자연에 존재하는 발화 신호는 기계 신호로 양자화하는 과정에서 1초에 몇 개의 샘플을 획득할 것인지의 Sample Rate(이하 SR)과 샘플을 몇 가지 수로 나타낼 것인지의 Bit Rate(이하 BR)로 2가지 변수를 가진다.

TTS 합성 분야에서 SR은 과거 16kHz부터, 이후 22.05kHz와 24kHz, 현재 32kHz, 44.1kHz, 48kHz까지 꾸준히 증가해왔다. Nyquist 이론에 근거하면 SR의 절반이 획득할 수 있는 주파대역의 최대치이므로, TTS는 과거 최대 8khz에서 24khz까지 점점 더 높은 주파대역을 복원할 수 있게 되었다.

**Intermediate representation**

TTS가 처음부터 높은 SR의 음성을 생성할 수 없었던 이유는, 음소의 발화 시간은 대략 10~50ms으로 1초에 20~50여개 정도이지만, 음성은 1초에 2만에서 4만여개 프레임으로 1k배 정도의 길이 차를 보이기에, highly contextual feature인 텍스트로부터 sparse feature인 음성 사이의 관계성을 학습시키기 어려웠기 때문이다.

이를 해결하기 위해 TTS 모델은 Spectral feature를 중간 매개로 두고, 텍스트에서 spectral feature을 합성하는 acoustic 모델과 spectral feature로부터 음성을 복원하는 vocoder 모델 2단계 구조를 구성하기 시작했다.

Spectral feature로는 대체로 Short-time Fourier Transform(이하 STFT)로 구해진 fourier feature의 magnitude 값(이하 power spectrogram)을 활용했다. 오픈소스 TTS 구현체에서는 주로 12.5ms 주기마다 50ms 정도의 음성 세그먼트를 발췌하여 주파 정보로 변환하였다. 이렇게 되면 spectral feature는 1초에 80개 정도의 프레임을 가지고, 텍스트에서 대략 2~4배, 음성까지 250~300배 정도로 구성된다.

Fourier feature를 활용할 수 있었던 이유는 \
1.) 음성의 발화 신호가 기본 주파수(F0)와 풍부한 배음(harmonics)로 구성되기에 fourier transform을 통해 각 주파대역별 세기를 나타내는 power spectrogram으로 표현하더라도 정보 유실이 크지 않았고 \
2.) Source-filter 이론에 근거하였을 때 발화 신호 중 발음 성분이 spectral magnitude의 형태(filter)에 상관관계를 가지고, 텍스트로부터 발음 정보를 만드는 것이 음성 신호를 만드는 것보다 상대적으로 쉬운 문제였기 때문이다.

그럼에도 filter에 직접 대응 가능한 quefrency 영역의 cepstral feature를 사용하지 않은 이유는, 기본 주파수 등의 정보 손실이 커 음성 신호로 복원하는 보코더의 문제 난이도를 어렵게 했기 때문이다.

**Log-scale, Mel-filter bank**

{{< figure src="/images/post/surveytts/powerspec.png" width="100%" caption="Figure 1: Power spectrogram" >}}

대체로 오픈소스 TTS 구현체에서는 STFT의 frequency bins를 1024개 혹은 2048개를 활용한다. 이때 TTS 모델이 합성해야 하는 프레임당 벡터의 길이는 spectral feature 중 허수 반전을 제외한 513개 혹은 1025개이다.

인간의 청각 체계는 음의 세기와 높낮이에 모두 log-scale로 반응한다. 신호의 세기가 N배 커지더라도, 실제로는 logN 정도로 인식하는 것이다. 이를 반영하여 인간이 실제로 듣는 신호의 세기와 높낮이를 강조하기 위해 TTS에서는 power spectrogram (linear spectrogram)을 곧장 활용하기보다는 주파대역의 인지적 선형화를 위해 filterbank를 취하고, 세기의 인지적 선형화를 위해 값에 log를 취한다.

filterbank는 주로 2가지를 활용하는 듯하다. 가장 많이 쓰이는 Mel-scale filterbank와 LPCNet[[git:xiph/LPCNet](https://github.com/xiph/LPCNet)] 등에서 간간이 보이는 Bark-scale filterbank이다. 대체로 오픈소스 TTS 구현체에서는 0Hz ~ 8kHz의 영역을 나눈 STFT 기존 513개~1025개 frequency bins를 80개~100개로 축약하는 mel-scale filterbank를 활용하는 편이다.

Mel-scale filterbank를 활용하는 경우를 log-Mel scale spectrogram이라고 하여, 간략히 mel-spectrogram이라 일컫는다.

**Vocoding**

보코더는 음성을 압축/복원하는 기술을 통칭한다. TTS에서는 algorithmic 하게 구해진 spectrogram(mel-scale)을 음성으로 복원하는 모델을 이야기한다.

단순히 STFT만을 취했다면 발화 신호 특성상 iSTFT만으로도 충분한 음성을 복원해낼 수 있지만, mel-spectrogram 변환 과정에서 \
1.) 주파대역별 세기를 측정하기 위해 실-허수 신호를 실수 신호로 축약하는 Absolute 연산 \
2.) 500~1000여개 bins를 80~100개로 압축하는 filter bank 연산의 2가지 손실 압축을 거치기에 algorithmic한 복원에는 한계가 존재한다.

Tacotron[[arXiv:1703.10135](https://arxiv.org/abs/1703.10135)]에서는 power-spectrogram을 활용하여 filter bank 연산이 없었고, 허수부-phase 복원에는 griffin-lim 알고리즘을 활용하였다.

상용화하기 어려운 음질이었고, 부족한 주파대역과 허수(phase) 정보 복원을 위해 Tacotron2[[arXiv:1712.05884](https://arxiv.org/abs/1712.05884)]에서는 2016년 WaveNet[[arXiv:1609.03499](https://arxiv.org/abs/1609.03499)]을 별도의 경험적 보코더로 두어, mel-spectrogram에서 time-domain signal 복원하도록 학습하여 활용하였다.

**Bit-rate**

음성은 과거와 현재 크게 다르지 않게 16bit를 bitrate로 산정하여, 음성 신호를 대략 6만여개 실수로 양자화하였다.

WaveNet을 경량화한 WaveRNN[[arXiv:1802.08435](https://arxiv.org/abs/1802.08435)]에서는 신호 복원 문제를 6만여개 클래스의 분류 문제로 바꾸고자 했는데, 현실적으로 6만개 클래스를 분류하는 것에는 학습에 어려움이 있었다.

이를 위해 당시에는 시간축 신호 역시 청각 구조에 따라 인지적 선형화를 진행하며 16bit를 8bit로 2차 양자화하였다. 대체로 mu-law를 활용하였으며, 8bit 256개 클래스로 분류하는 보다 쉬운 문제로 치환하였다.

하지만 mu-law 역시 손실 압축이기 때문에, 복원된 8bit 음성을 algorithmic 하게 16bit로 복원하는 과정에서 배경 노이즈가 생성되는 이슈가 있었다.

이는 이후 GAN 기반 신호 복원 방법론인 MelGAN[[arXiv:1910.06711](https://arxiv.org/abs/1910.06711)] 등이 등장하며 bitrate 상관없이 [-1, 1]의 실수 범위 신호를 직접 복원하였다.

**Now-on**

근래에는 24khz, 32khz, 48khz의 SR과 16bit BR의 데이터셋을 주로 활용하고 있으며, 대체로 1024bins/12.5ms(or 256frame)/50ms(or 1024frame)의 STFT, 80~100bins logMel-scale spectrogram을 활용하는 듯하다. [[git:seungwonpark/melgan](https://github.com/seungwonpark/melgan), [git:jik876/hifi-gan](https://github.com/jik876/hifi-gan)] 아무래도 Tacotron2의 영향이지 않을까 싶다. [[git:NVIDIA/tacotron2](https://github.com/NVIDIA/tacotron2)]

이외로 preemphasis 필터를 거치거나, 기준 세기를 잡아 amplitude 영역의 주파정보를 decibel 단위로 변환하기도 하고, 기준 세기를 토대로 [-1, 1] 범위로 값을 scaling 하기도 한다. [[git:keithito/tacotron](https://github.com/keithito/tacotron)]

---

{{< details summary="TODO" >}}

**TODO - Autoregressive TTS**

Bahdanau
- Neural Machine Translation by Jointly Learning to Align and Translate, Bahdanau et al., 2014. https://arxiv.org/abs/1409.0473

Tacotron - location sensitive
- Tacotron: Towards End-to-End Speech Synthesis, Wang et al., 2017. https://arxiv.org/abs/1703.10135

DCTTS - Guided attention loss
- Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention, Tachibana et al., 2017. https://arxiv.org/abs/1710.08969

TransformerTTS - 
- Neural Speech Synthesis with Transformer Network, Li et al., 2019. https://arxiv.org/abs/1809.08895


Forward attention
- Forward Attention in Sequence-to-sequence Acoustic Modeling for Speech Synthesis, Zhang et al., 2018. https://arxiv.org/abs/1807.06736

Dynamic convolution attention
- Dynamic Convolution: Attention over Convolution kernels, Chen et al., 2019. https://arxiv.org/abs/1912.03458

Flowtron

NAT

---

**TODO - Parallel TTS**

VQ-TTS

EATS

GAN-TTS, SED-TTS

Parallel-Tacotron 1/2

Duration modeling
- FastSpeech: Fast, Robust and Controllable Text to Speech, Ren et al., 2019.
- SpeedySpeech

TalkNet

ParaNet

Glow-TTS, VITS, Flow-TTS

AlignTTS
- Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks, Graves et al., 2006.

JDI-T

**Reference**
- A Survey on Neural Speech Synthesis, Tan et al., 2021. [[arXiv:2106.15561](https://arxiv.org/abs/2106.15561)]

{{< /details >}}



