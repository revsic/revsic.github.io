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

Hi, I'm Young Joong Kim, a TTS researcher at [LionRocket](https://lionrocket.ai).

안녕하세요, [라이언로켓](https://lionrocket.ai)에서 TTS 리서처를 맡고 있는 김영중입니다.

I'm the research team leads, part of Speech Synthesis, and also be in charge of head researcher.

저는 음성 합성 파트의 연구팀장으로, 실질적인 연구와 총괄을 겸임하고 있습니다.

I'm interested in Generative models for TTS system, and also following the other recent papers.

생성 모델과 TTS 시스템에 관심이 있으며, 다른 분야의 최신 논문들도 찾아보고 있습니다.

**Works**

- TTS Researcher at [LionRocket](https://lionrocket.ai) \
(2019.09 ~)

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

- CULICULI [GIT:private, ORG:[lionrocket-inc](https://github.com/lionrocket-inc/)], 2020.07.10 ~ 2020.07.16 \
: *CUDA Lib for LionRocket*

---

- LR_TTS [GIT:private, ORG:[lionrocket-inc](https://github.com/lionrocket-inc/)], 2019.09 ~ \
: *PyTorch implementation of TTS base modules*

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
