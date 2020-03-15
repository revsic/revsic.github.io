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

Hi, I'm Young Joong Kim, a TTS Machine learning researcher at [LionRocket](https://lionrocket.ai).

안녕하세요, [라이언로켓](https://lionrocket.ai)에서 TTS 머신러닝 리서치를 맡고 있는 김영중입니다.

I'm working on the Text-To-Speech deep learning system and optimizing the inference module by CUDA C++.

저는 Text-To-Speech 딥러닝 시스템 관련 연구를 진행하고, CUDA C++를 통해 inference 모듈을 최적화하는 일을 하고 있습니다.

I'm interested in Bayesian methodology and TTS system, and also following the recent papers from Reddit, etc.

베이지안 방법론과 TTS 시스템에 관심이 있으며, 레딧 등을 통해 최신 논문들도 찾아보고 있습니다.

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

- Behavior based Malware Detection Using Branch Data [[GIT](https://github.com/revsic/tf-branch-malware)] \
: *Classify malware from benign software using branch data via LSTM based on Tensorflow*

브랜치 데이터를 통한 행위 기반 멀웨어 탐지 기법 연구 \
Skills: C++, Windows Internal, Sandbox, PE header, Python, Tensorflow

VEH 기반 Branch Tracer를 구현한 후, DLL Injection 방식을 통해 sandbox 환경에서 멀웨어와 일반 소프트웨어의 브랜치 데이터를 축적, 텐서플로 모듈을 통해 딥러닝 기반 탐지 모델을 개발하였습니다.

Sandbox 환경 내에서 MSR을 사용할 수 없어 VEH를 통해 branch tracer를 직접구현해야 했고, 분기문을 검색하기 위해 디스어셈블러의 일부를 직접 구현해야 하면서 기술적 어려움을 겪었습니다. 이는 후에 인텔 매뉴얼을 참고하며 dll을 완성하였고, 이후에는 이를 발전시켜 VEH 기반 DBI를 구현할 수 있었습니다. ([cpp-veh-dbi](https://github.com/revsic/cpp-veh-dbi)).

딥러닝 모델은 Embedding + LSTM + FC x 2 의 간단한 시퀸스 모델을 이용하였고, 결과 88% 정도의 정확도를 확인할 수 있었습니다.

현재는 Self-Attention 등의 모듈을 통해 메소드의 종류와 소프트웨어 부류의 관계, 메소드 임베딩 벡터 탐색 등 interpretability 측면에서 실험해 보고 있습니다.

---

- AlphaZero Connect6 [[GIT](https://github.com/revsic/AlphaZero-Connect6)] \
: *AlphaZero training framework for game Connect6 written in Rust with C++, Python interface.*

Rust로 작성된 육목 게임를 위한 AlphaZero 알고리즘 구현 \
Skills: Rust, C++, Python, Tensorflow, LibTorch, Azure CI

Rust로 게임 육목을 구현한 후 이의 A.I. 플레이어 구현을 위해 AlphaZero 알고리즘을 함께 구현한 프로젝트입니다. AlphaZero 알고리즘은 Rust로 구현하였고, Neural Network가 이용되는 Evaluation 부분을 Callback 인터페이스로 구성하여, 다른 언어로 구현된 딥러닝 모델을 이용할 수 있게 하였습니다.

가장 먼저 Python Tensorflow 기반으로 Interface를 구성하여 진행해 보았습니다. 하지만 Single GPU에서 여러 개의 게임을 병렬로 실행시켜 데이터를 쌓기에는 추론 속도가 느렸고, Python 속도의 한계라 판단하여 C++ 기반의 프레임워크를 이용해 보았습니다.

C++ LibTorch 기반의 Interface를 구성하여 진행해 보았습니다. 하지만 여전히 느린 속도에 병목 지점을 확인해 보았고, GPU 연산은 같은 시간대에 하나의 프로세서에서만 사용할 수 있는 time slicing 방식의 scheduling을 지원하고 있었기에, 추후 computational power가 넉넉할 때 다시 도전해 보기로 기약하며 결과를 확인하지 못하고 마무리 지었습니다.

---

- RosettaStone [[GIT](https://github.com/utilForever/RosettaStone)] \
: *C++ implementation of game 'Hearthstone' as training environment and A.I. for future work.*

게임 하스스톤을 위한 C++ 기반 강화학습 환경 구현 \
Skills: C++

게임 하스스톤은 온라인 카드 수집 게임으로, 사용자가 서로 다른 속성을 가진 카드 중에서 주어진 수만큼을 선택, 상대방과 돌아가며 카드를 내고, 상호작용하여 상대방의 체력 수치를 0 이하로 만드는 사람이 이기는 게임입니다.

현재 하스스톤은 제작사에서 공개한 AI 개발용 시뮬레이션 환경이 존재하지 않아, C++ 기반으로 학습 환경을 직접 구현해 보기로 하였습니다.

게임 시뮬레이터를 만들기 위해서는 6천 장 가까이 되는 카드를 일일이 구현해야 하고, 상호작용에 있어서 실제 게임과 동일하게 구성해야 했습니다. 저희는 게임을 진행 순서와 상호작용, 카드 3가지로 나누어 보았고, 저는 그중에서 진행 순서 부분을 구현하게 되었습니다.

현재에도 게임이 업데이트되면서 추가 카드와 상호작용이 나올 경우 해당 분야 팀에서 구현을 하고 있고, 기본 덱이 어느 정도 완성되어 DQN 등의 알고리즘을 학습해 보고 있습니다.

**Paper Implementation**
1. tf-neural-process [[GIT](https://github.com/revsic/tf-neural-process)] [arxiv: [NP](https://arxiv.org/abs/1807.01622), [CNP](https://arxiv.org/abs/1807.01613), [ANP](https://arxiv.org/abs/1901.05761)] \
: *Neural process, Conditional Neural Process, Attentive Neural Process*

2. tf-began [[GIT](https://github.com/revsic/tf-began)] [[arXiv](https://arxiv.org/abs/1703.10717)] \
: *BEGAN: Boundary Equilibrium Generative Adversarial Networks*

3. tf-vanilla-gan [[GIT](https://github.com/revsic/tf-vanilla-gan)] [[arXiv](https://arxiv.org/pdf/1406.2661.pdf)] \
: *Generative Adversarial Networks*

4. tf-dcgan [[GIT](https://github.com/revsic/tf-dcgan)] [[arXiv](https://arxiv.org/abs/1511.06434)] \
: *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*

이 외의 활동은 [work list](../blog/worklist)에서 확인 가능합니다. \
Other activities can be found in [work list](../blog/worklist).
