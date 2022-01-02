---
title: "About"
date: 2019-10-29T13:49:23+06:00
draft: false

# image
image: "images/author.jpg"

# meta description
description: "author page"

# type
type : "about"
---

Hi, I'm Young Joong Kim, a Research team lead at [LionRocket](https://lionrocket.ai). \
I lead both research teams of speech and video synthesis, and also be in charge of speech team head researcher. \
I'm interested in Generative models, and also following the other recent papers.

안녕하세요, [라이언로켓](https://lionrocket.ai)에서 연구팀장을 맡은 김영중입니다. \
저는 음성, 영상 합성팀의 매니징과 음성 연구원의 업무를 겸임하고 있습니다. \
생성 모델 전반을 관심 있게 보고 있습니다.

**Works**

- 음성 합성 연구원, TTS Researcher at [LionRocket](https://lionrocket.ai) \
(2019.09. ~)

- 연구팀장, Research Team Lead at [LionRocket](https://lionrocket.ai) \
(2021.04. ~)

**Education**

- [한양대학교](https://www.hanyang.ac.kr) 컴퓨터소프트웨어학부  \
Department of Computer Software Engineering at [Hanyang University](https://www.hanyang.ac.kr) \
(2018.03 ~)

- [KITRI BoB](https://www.kitribob.kr/), 5기 취약점 분석 트랙 \
5th Vulnerability Analysis Track \
(2015.03 ~ 2017.02)

- [선린인터넷고등학교](http://sunrint.hs.kr/) 정보통신과 \
Department of Information and Communication Technology at [Sunrin Internet High School](http://sunrint.hs.kr) \
(2015.03 ~ 2017.02)

**Awards**

- KISA, SW 보안 경진대회, 어플리케이션 보안 부문 2위, 행정자치부 장관상 \
KISA, Software Contest, Application Security Section 2nd Prize, Minister of the Interior Award \
2016.09

**Presentation**

- GP to NP: Gaussian process and Neural Process [[SlideShare](https://www.slideshare.net/YoungJoongKim1/gp-to-np-gaussian-process-and-neural-process-230289387)] \
[A.I.U 1st Open AI Conference](https://festa.io/events/288), 2019.05 \
Gaussian Process 부터 Neural Process 까지 확률적 프로세스와 뉴럴넷이 만나가는 과정

- <하스스톤> 강화학습 환경 개발기 \
\<Hearthstone\> Developing Environment for RL \
[Nexon Developers Conference 2019](https://ndc.nexon.com/main), Team [RosettaStone](https://github.com/utilForever/RosettaStone), 2019.04 \
하스스톤 강화학습 환경 구성을 위한 C++ 기반 시뮬레이터 개발 일기

- Hearthstone++: Hearthstone simulator with reinforcement learning \
[Deep Learning Camp Jeju](http://jeju.dlcamp.org/2018/), 2018.07 \
하스스톤 강화학습 가능성에 대한 간단한 연구

- Behavior based Malware Detection Using Branch Data [[SlideShare](https://www.slideshare.net/YoungJoongKim1/behavior-based-malware-detection-using-branch-data-230288166)]\
[CodeGate 2017 Junior](https://www.codegate.org/), 2017.04 \
브랜치 데이터와 딥러닝을 통해 여러 실행 파일 중에서 멀웨어를 분류해 내는 실험

**Project Overview**

- (가칭) Stable TTS System (2021.09. ~ 2021.12.) \
: TTS 합성 실패 방지의 이론적 해결책에 대한 연구

- (가칭) TTS Latent system (2021.04. ~ 2021.08.) \
: Variance-bias trade-off를 통해 합성 품질을 안정화 하고, internal feature을 hierarchy에 따라 역할과 목적을 명시화하여 non-parallel 데이터에서 다국어, 감정 등 unseen property에 대한 일반화 성능을 높히는 연구

- (가칭) Semi-Autoregressive TTS (2020.12. ~ 2021.04.) \
: Parallel TTS 모델의 합성 속도와 엔저니어링 단순화의 이점, Autoregressive TTS의 음질상 이점 사이 Trade-off에 대한 연구

---

- CULICULI [GIT:private, [lionrocket-inc](https://github.com/lionrocket-inc/)], 2020.07.10 ~ 2020.07.16 \
: *CUDA Lib for LionRocket*

C++ CUDA Native를 활용하여 딥러닝 추론 속도를 10배 가량 가속화한 프레임워크 \
Skills: C++, CUDA

음성 합성 파이프라인의 추론 가속화를 위해 C++ CUDA Native를 활용하여 10배가량 합성 시간을 단축시킨 프로젝트입니다. C++과 CUDA를 통해 기본적인 Tensor 객체와 BLAS(Basic Linear Algebra Subroutines)를 구성하고, 합성 속도를 최적화한 후, [PyBind](https://pybind11.readthedocs.io/en/stable/)를 통해 python 인터페이스를 제공하였습니다.

(Details) \
당시 TTS 모델에는 음성의 길이에 합성 시간이 비례하는 문제가 있었고, 단위 시간을 줄여 거의 실시간에 가까운 합성 속도를 구성할 수 있어야 했습니다. 이를 위해 C++로 BLOB-Shape Tuple 형태의 Tensor 객체를 구축하고, template meta programming을 통해 이를 CUDA Native에서도 활용할 수 있게 두었습니다.

BLAS 구현과 POC 이후 병목이 메모리 할당에 있음을 확인하여, 메모리 풀과 CUDA API를 활용하지 않는 자체적인 메모리 할당 방식을 구성, 대략 5~7배의 속도 향상을 확인할 수 있었습니다.

이렇게 만들어진 프레임워크를 팀에서 활용하고자 했고, LR_TTS에서 학습된 체크포인트 파일과 파이썬 인터페이스로 실행 가능하도록 [PyBind](https://pybind11.readthedocs.io/en/stable/)를 차용해보았습니다.

---

- LR_TTS [GIT:private, [lionrocket-inc](https://github.com/lionrocket-inc/)], 2019.09 ~ \
: *PyTorch implementation of TTS base modules*

음성 데이터 전처리, 모델 구현, 학습, 데모, 패키징, 배포까지의 파이프라인을 구성한 프레임워크 \
Skills: Python, PyTorch, Librosa, Streamlit, Tensorboard

음성 합성팀의 통합 연구 환경을 위한 플랫폼 개발 프로젝트입니다. 당시 PyTorch에는 Keras나 Lightning과 같이 단순화된 프레임워크가 부재했기에 데이터 생성부터 연구, 개발, 학습, 패키징, 배포, 데모 등 일련의 과정을 프로세스화 하고 코드 재사용성을 극대화하여 적은 리소스로 연구자가 부담없이 배포가 가능하도록 구성했습니다.

(Details) \
자사 내의 데이터 전처리 구조를 단순화하고, 모든 학습이 고정된 프로토콜 내에서 가능하도록 모델 구조와 콜백 함수를 추상화하여 연구 프로세스를 정리했습니다. 또한 패키징과 배포의 단순화를 위해 모델 구조와 하이퍼파라미터를 분리, 각각을 고정된 프로토콜에 따라 저장, 로딩하는 모든 과정이 자동화될 수 있도록 구성했습니다.

개발 중 UnitTest와 CI를 도입해보았지만, 딥러닝 모델의 테스트 방법론이 일반적인 소프트웨어 테스트 방법론과는 상이한 부분이 존재했고, 끝내 테스트가 관리되지 않아 현재는 테스트를 제거한 상태입니다.

CI의 경우에는 이후 PR 생성에 따라 자동으로 LR_TTS의 버전 정보를 생성하고, on-premise framework에 모델만 분리, 배포를 자동화 하는 방식으로 활용 중입니다.

---

- Behavior based Malware Detection Using Branch Data [[GIT](https://github.com/revsic/tf-branch-malware)], 2019.08. \
: *Classify malware from benign software using branch data via LSTM based on Tensorflow*

브랜치 데이터를 통한 행위 기반 멀웨어 탐지 기법 연구 \
Skills: C++, Windows Internal, PE, Cuckoo Sandbox, Python, Tensorflow

[VEH](https://docs.microsoft.com/en-us//windows/win32/debug/vectored-exception-handling)를 기반으로 분기구문(branch instruction)을 추적하는 [Branch Tacer](https://github.com/revsic/BranchTracer)를 구현한 후, DLL Injection 방식을 통해 보안 가상 환경(sandbox)에서 멀웨어와 일반 소프트웨어의 분기 정보(branch data)를 축적, [딥러닝 기반 탐지 모델](https://github.com/revsic/tf-branch-malware)을 개발하였습니다.

(Details) \
Sandbox 환경 내에서는 MSR을 사용할 수 없어 VEH를 통해 branch tracer를 직접 구현해야 했고, 분기문 탐색을 위해 디스어셈블러의 일부를 직접 구현하면서 기술적 어려움을 겪었습니다. 이는 후에 인텔 매뉴얼을 참고하며 tracer를 완성하였고, 이후 이를 발전시켜 VEH 기반의 DBI(Dynamic Binary Instrumentation)[[GIT:cpp-veh-dbi](https://github.com/revsic/cpp-veh-dbi)] 도구를 구현할 수 있었습니다.

딥러닝 모델은 LSTM 기반의 간단한 시퀸스 모델을 이용하였고, 결과 88% 정도의 정확도를 확인할 수 있었습니다.

이는 당시 논문의 형태로 정리되어 [정보과학회](https://www.kiise.or.kr/) 2017년 한국컴퓨터종합학술대회 논문집[[PAPER](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE07207863)]에 고등학생 부문으로 기재되었습니다. 

**Model Implementation**
- tf-mlptts [[GIT](https://github.com/revsic/tf-mlptts)], 2021.09. \
: *Tensorflow implementation of MLP-Mixer based TTS.*

- jax-variational-diffwave [[GIT](https://github.com/revsic/jax-variational-diffwave)], [[arXiv:2107.00630](https://arxiv.org/abs/2107.00630)], 2021.09. \
: *Variational Diffusion Models*

- tf-glow-tts [[GIT](https://github.com/revsic/tf-glow-tts)] [[arXiv:2005.11129](https://arxiv.org/abs/2005.11129)], 2021.07. \
: *Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search*

- tf-survae-flows [[GIT](https://github.com/revsic/tf-survae-flows)], [[arXiv:2007.023731](https://arxiv.org/abs/2007.02731)], 2021.05. \
: *SurVAE Flows: Surjections to Bridge the Gap between VAEs and Flows*

- tf-diffwave [[GIT](https://github.com/revsic/tf-diffwave)] [[arXiv:2009.09761](https://arxiv.org/abs/2009.09761)], 2020.10. \
: *DiffWave: A Versatile Diffusion Model for Audio Synthesis, Zhifeng Kong et al., 2020.*

- Rewriting-A-Deep-Generative-Models [[GIT](https://github.com/revsic/Rewriting-A-Deep-Generative-Models)] [[arXiv:2007.15646](https://arxiv.org/abs/2007.15646)], 2020.09. \
: *Rewriting a Deep Generative Model, David Bau et al., 2020.* 

- tf-alae [[GIT](https://github.com/revsic/tf-alae)] [[arXiv:2004.04467](https://arxiv.org/abs/2004.04467)], 2020.09. \
: *Adversarial Latent Autoencoders, Stanislav Pidhorskyi et al., 2020.*

- tf-neural-process [[GIT](https://github.com/revsic/tf-neural-process)] [arxiv: [NP](https://arxiv.org/abs/1807.01622), [CNP](https://arxiv.org/abs/1807.01613), [ANP](https://arxiv.org/abs/1901.05761)], 2019.05 \
: *Neural process, Conditional Neural Process, Attentive Neural Process*

- tf-vanilla-gan [[GIT](https://github.com/revsic/tf-vanilla-gan)] [[arXiv:1406.2661](https://arxiv.org/pdf/1406.2661.pdf)], 2018.01. \
: *Generative Adversarial Nets, Ian J. Goodfellow et al., 2014.*

이 외의 활동은 [work list](../blog/worklist)에서 확인 가능합니다. \
Other activities can be found in [work list](../blog/worklist).
