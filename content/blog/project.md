---
title: "Project Overview"
date: 2022-08-13T16:34:15+09:00
draft: false

# post thumb
image: "images/post/project/head.jpg"

# meta description
description: "Project Overview"

# taxonomies
categories: 
  - "Portfolio"
tags:
  - "Portfolio"

# post type
type: "featured"
---

**Vision**

- Object Detection/Recognition/Search, 2022.08. \
R&R: 1인 연구, 영상 합성 연구원

- Object Classification/Recommendation, 2022.07. \
R&R: 1인 연구, 영상 합성 연구원

- Image Generation, 2022.06. \
R&R: 1인 연구, 영상 합성 연구원

- Lip-sync Video Synthesis, 2022.01. \
R&R: 영상 합성 프로젝트 매니저

---

**Speech**

- Stable TTS, 2021.09. ~ 2021.12. \
: TTS 합성 실패 방지의 이론적 해결책에 관한 연구

R&R: 1인 연구, 음성 합성 연구원, 실험 실패 방지를 위한 연구 수행

근래의 대부분 딥러닝 모델은 BatchNorm이나 InstanceNorm을 활용합니다. 이 중 BatchNorm은 학습 과정에서 추정한 이동 통계량을 기반으로 표준화를 진행합니다. 만약 학습에 활용한 데이터의 양이 충분하지 않아 통계치가 일반화되지 않았다면 miss-normalization 문제가 발생할 수 있습니다.

저량의 데이터로 학습된 합성 모델에서 음성이 오합성 되는 이슈가 있었고, 분석 결과 BatchNorm의 miss-normalization에 의한 feature map의 variance exploding 현상을 원인으로 확인하였습니다.

이를 해결하기 위해 RescaleNet[[NeurIPS2020](https://papers.nips.cc/paper/2020/file/9b8619251a19057cff70779273e95aa6-Paper.pdf)], LayerScale[[arXiv:2103.17239](https://arxiv.org/abs/2103.17239v2)], InstanceNorm 등으로 대체하는 연구를 진행하였습니다.

---

- Latent system, 2021.04. ~ 2021.08. \
: *non-parallel 데이터와 unseen property의 일반화 가능성에 관한 연구*

R&R: 1인 연구, 음성 합성 연구원, 다국어 모델 개발을 위한 연구 수행

음성은 크게 발화자/언어/비언어 표현 3가지 관점에서 관찰할 수 있습니다. 이중 각 도메인의 클래스 간 모든 조합을 데이터로 구성하는 것을 parallel data, 일부 케이스가 비는 것을 non-parallel data라고 할 때, non-parallel 환경에서 문장 내 화자와 언어 정보를 분리하는 것은 natural하게 이뤄질 수 없습니다.

- ex. [인물A/B, 영어/한글], parallel: 인물A/한글, 인물A/영어, 인물B/한글, 인물B/영어
- natural: 케이스가 비는 경우, 별도의 장치 없이 화자와 언어를 조건화하는 것만으로는 unseen pair의 합성 품질을 보장할 수 없습니다.

따라서 non-parallel 환경에서 다화자-다국어 음성 합성 모델을 개발하는 경우, 특정 화자에서 관측되지 않은 언어 정보, unseen property에 대한 일반화가 이뤄질 수 있어야 합니다.

Latent System 연구에서는 VAE와 GAN 등 방법론을 통해 Latent variable을 도입하고, 정보의 흐름을 보다 명확히 관리하는 것을 목표로 합니다. CLUB[[arXiv:2006.12013](https://arxiv.org/abs/2006.12013)]을 활용한 국소-전역부의 잠재 변수 분리, CycleGAN[[arXiv:1703.10593](https://arxiv.org/abs/1703.10593)]을 활용한 unseen-property 일반화 등을 가설로 연구를 수행하였습니다.

다음은 당시 모델로 만들었던 프로토타입 영상입니다.

- [youtube:Lionrocket](https://www.youtube.com/watch?v=38LrO_cbAyU)

---

- Semi-Autoregressive TTS, 2020.12. ~ 2021.04. \
: *합성 속도와 음질상 이점의 Trade-off에 관한 연구*

R&R: 음성 합성 연구원, 베이스라인 개선 실험 수행

TTS 모델은 Autoregressive(이하 AR) 모델과 Duration 기반의 Parallel(이하 PAR) 모델로 나뉩니다. AR 모델은 대체로 합성 속도가 음성의 길이에 비례하여 느려지지만 전반적인 음질 수준이 높고, PAR 모델은 상수 시간에 가까운 합성 속도를 가지지만 전반적으로 노이즈 수준이 높은 편입니다. 

Semi-Autoregressive TTS 연구는 이 둘을 보완하기 위한 연구입니다. AR TTS의 병목은 대부분은 AR 방식의 Alignment에서 오기에, Alignment는 Duration 기반의 PAR 모델을 따르고, 이후 Spectrogram 생성은 Autoregression하는 방식의 가설로 삼았습니다. 이는 DurIAN[[arXiv:1909.01700](https://arxiv.org/abs/1909.01700)], NAT[[2010.04301](https://arxiv.org/abs/2010.04301)]와 유사합니다.

이후 추가 개선을 거쳐 실시간에 가까운 AR 모델을 개발하였지만, 음질의 중요성이 높아지며 추가 개선 및 배포가 보류된 프로젝트입니다.

---

- TTS Baseline, 2019.09. ~ 2020.10. \
: *Text-to-Speech 음성 합성 모델 베이스라인 선정에 관한 연구*

R&R: 음성 합성 연구원, 오픈소스 검토, 논문 구현, Ablation

TTS 모델의 베이스라인 선정에 관한 연구입니다. Autoregressive 모델인 Tacotron[[arXiv:1703.10135](https://arxiv.org/abs/1703.10135)]부터 Duration 기반의 parallel 모델인 FastSpeech2[[arXiv:2006.04558](https://arxiv.org/abs/2006.04558)] 등을 폭넓게 검토하였습니다. 검토 과정에서 어떤 백본을 썼을 때 발음이나 음질 오류가 줄어드는지 검토하고, Duration을 어떤 모델을 통해 추정할지, Joint training이 가능한지를 연구하였습니다.

Acoustic 모델이 완료된 후에는 Vocoder 군에서 Autoregressive 모델인 WaveNet[[arXiv:1609.03499](https://arxiv.org/abs/1609.03499)], WaveRNN[[arXiv:1802.08435](https://arxiv.org/abs/1802.08435)] LPCNet[[arXiv:1810.11846](https://arxiv.org/abs/1810.11846)]과 Parallel 모델인 MelGAN[[arXiv:1910.16711](https://arxiv.org/abs/1910.06711)] 등을 검토하였습니다. 이후 LPCNet에서 영감을 받아 Source-filter 기반의 방법론을 GAN 기반의 Parallel 모델에 적용하여 음질 개선이 이뤄질 수 있는지 연구하였습니다.

연구된 베이스라인은 TTS 서비스인 [On-air studio](https://onairstudio.ai/)에서 활용하고 있습니다.

{{< details summary="다음은 그 외 사이드 프로젝트로 구현한 TTS 모델입니다.">}}
- torch-diffusion-wavegan [[GIT](https://github.com/revsic/torch-diffusion-wavegan)], 2022.03. \
: *Parallel waveform generation with DiffusionGAN, Xiao et al., 2021.*

- torch-tacotron [[GIT](https://github.com/revsic/torch-tacotron)], 2022.02. \
: *PyTorch implementation of Tacotron, Wang et al., 2017.* 

- tf-mlptts [[GIT](https://github.com/revsic/tf-mlptts)], 2021.09. \
: *Tensorflow implementation of MLP-Mixer based TTS.*

- jax-variational-diffwave [[GIT](https://github.com/revsic/jax-variational-diffwave)], [[arXiv:2107.00630](https://arxiv.org/abs/2107.00630)], 2021.09. \
: *Variational Diffusion Models*

- tf-glow-tts [[GIT](https://github.com/revsic/tf-glow-tts)] [[arXiv:2005.11129](https://arxiv.org/abs/2005.11129)], 2021.07. \
: *Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search*

- tf-diffwave [[GIT](https://github.com/revsic/tf-diffwave)] [[arXiv:2009.09761](https://arxiv.org/abs/2009.09761)], 2020.10. \
: *DiffWave: A Versatile Diffusion Model for Audio Synthesis, Zhifeng Kong et al., 2020.*

{{< /details >}}

---

**Engineering**

- face_provider [GIT:[lionrocket-inc](https://github.com/lionrocket-inc/)/private], 2022.06 \
: *All-in-one Face generation API*

얼굴 인식, 검색, 합성, 분류, 추천 목적 통합 서비스 지원 프레임워크 \
Skills: Python, PyTorch, dlib, opencv, FAISS \
R&R: 1인 개발

통합 얼굴 이미지 지원 프레임워크입니다. 이미지 내 얼굴 탐지를 시작으로 정렬, 인식, 분류, 벡터 데이터베이스에서의 검색과 추천을 지원합니다.

얼굴 탐지와 인식 과정에는 입력 이미지의 회전량에 따라 인식 성능이 떨어지는 문제가 있었고, 이를 보정하기 위해 두상의 회전량을 추정하여 이미지를 정면으로 정렬하거나, 인식이 불가능한 이미지를 사전에 고지할 수 있게 구성하였습니다.

이후 검색과 분류, 추천 과정이 실시간으로 이뤄져야 한다는 기획팀의 요청이 있었고, 벡터 검색 과정은 MetaAI의 벡터 검색 시스템 [FAISS](https://github.com/facebookresearch/faiss)를 활용하여 최적화를 진행하였습니다. 얼굴형에 관한 초기 분류 모델은 [dlib](http://dlib.net/)의 Facial Landmark를 기반으로 작동하였으나, [dlib](http://dlib.net/)은 실시간 구성이 어렵다는 문제가 있었고, 추후 [Mediapipe](https://google.github.io/mediapipe/) 교체를 고려하고 있습니다.

---

- CULICULI [GIT:[lionrocket-inc](https://github.com/lionrocket-inc/)/private], 2020.07.10 \
: *CUDA Lib for LionRocket*

C++ CUDA Native를 활용하여 딥러닝 추론 속도를 10배 가량 가속화한 프레임워크 \
Skills: C++, CUDA, Python, PyBind \
R&R: 1인 개발

음성 합성 파이프라인의 추론 가속화를 위해 C++ CUDA Native를 활용하여 10배가량 합성 시간을 단축시킨 프로젝트입니다. C++과 CUDA를 통해 기본적인 Tensor 객체와 BLAS(Basic Linear Algebra Subroutines)를 구성하고, 합성 속도를 최적화한 후, [PyBind](https://pybind11.readthedocs.io/en/stable/)를 통해 python 인터페이스를 제공하였습니다.

당시 TTS 모델에는 음성의 길이에 합성 시간이 비례하는 문제가 있었고, 단위 시간을 줄여 거의 실시간에 가까운 합성 속도를 구성할 수 있어야 했습니다. 이를 위해 C++로 BLOB-Shape Tuple 형태의 Tensor 객체를 구축하고, 템플릿 프로그래밍을 통해 이를 CUDA Native에서도 활용할 수 있게 두었습니다.

BLAS 구현과 POC 이후 병목이 메모리 할당에 있음을 확인하여, 메모리 풀과 CUDA API를 활용하지 않는 자체적인 메모리 할당 방식을 구성, 대략 5~7배의 속도 향상을 확인할 수 있었습니다.

이렇게 만들어진 프레임워크를 팀에서 활용하고자 했고, LR_TTS에서 학습된 체크포인트를 파이썬 인터페이스로 실행 가능하도록 [PyBind](https://pybind11.readthedocs.io/en/stable/)를 활용하였습니다.

---

- LR_TTS [GIT:[lionrocket-inc](https://github.com/lionrocket-inc/)/private], 2019.09 \
: *PyTorch implementation of TTS base modules*

음성 데이터 전처리, 모델 구현, 학습, 데모, 패키징, 배포까지의 파이프라인을 구성한 프레임워크 \
Skills: Python, PyTorch, Librosa, Streamlit, Tensorboard \
R&R: 기획, 개발, 배포, 총책임

음성 합성팀의 통합 연구 환경을 위한 플랫폼 개발 프로젝트입니다. 당시 PyTorch에는 Keras나 Lightning과 같이 단순화된 프레임워크가 부재했기에 데이터 생성부터 연구, 개발, 학습, 패키징, 평가, 배포, 데모 등 일련의 과정을 프로세스화 하고 코드 재사용성을 극대화하여 적은 리소스로 연구자가 부담없이 배포가 가능하도록 구성했습니다.

자사 내의 데이터 전처리 구조를 단순화하고, 모든 학습이 고정된 프로토콜 내에서 가능하도록 모델 구조와 콜백 함수를 추상화하여 연구 프로세스를 정리했습니다. 또한 패키징과 배포의 단순화를 위해 모델 구조와 하이퍼파라미터를 분리, 각각을 고정된 프로토콜에 따라 저장, 로딩하는 모든 과정이 자동화될 수 있도록 구성했습니다.

개발 중 UnitTest와 CI를 도입해보았지만, 딥러닝 모델의 테스트 방법론이 일반적인 소프트웨어 테스트 방법론과는 상이한 부분이 존재했고, 끝내 테스트가 관리되지 않아 현재는 테스트를 제거한 상태입니다.

CI의 경우에는 이후 PR 생성에 따라 자동으로 LR_TTS의 버전 정보를 생성하고, on-premise framework에 모델을 자동으로 배포할 수 있도록 구성하였습니다.

---

- Behavior based Malware Detection Using Branch Data [[GIT](https://github.com/revsic/tf-branch-malware)], 2019.08. \
: *Classify malware from benign software using branch data via LSTM based on Tensorflow*

브랜치 데이터를 통한 행위 기반 멀웨어 탐지 기법 연구 \
Skills: C++, Windows Internal, PE, Cuckoo Sandbox, Python, Tensorflow \
R&R: 1인 연구

[VEH](https://docs.microsoft.com/en-us//windows/win32/debug/vectored-exception-handling)를 기반으로 분기구문(branch instruction)을 추적하는 [Branch Tacer](https://github.com/revsic/BranchTracer)를 구현한 후, DLL Injection 방식을 통해 보안 가상 환경(sandbox)에서 멀웨어와 일반 소프트웨어의 분기 정보(branch data)를 축적, [딥러닝 기반 탐지 모델](https://github.com/revsic/tf-branch-malware)을 개발하였습니다.

Sandbox 환경 내에서는 MSR을 사용할 수 없어 VEH를 통해 branch tracer를 직접 구현해야 했고, 분기문 탐색을 위해 디스어셈블러의 일부를 직접 구현하면서 기술적 어려움을 겪었습니다. 이는 후에 인텔 매뉴얼을 참고하며 tracer를 완성하였고, 이후 이를 발전시켜 VEH 기반의 DBI(Dynamic Binary Instrumentation)[[GIT:cpp-veh-dbi](https://github.com/revsic/cpp-veh-dbi)] 도구를 구현할 수 있었습니다.

딥러닝 모델은 LSTM 기반의 간단한 시퀸스 모델을 이용하였고, 결과 88% 정도의 정확도를 확인할 수 있었습니다.

이는 당시 논문의 형태로 정리되어 [정보과학회](https://www.kiise.or.kr/) 2017년 한국컴퓨터종합학술대회 논문집[[PAPER](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE07207863)]에 고등학생 부문으로 기재되었습니다. 
