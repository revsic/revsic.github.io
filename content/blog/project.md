---
title: "Project Overview"
date: 2022-08-13T16:34:15+09:00
draft: true

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

**Speech**

TODO

---

**Vision**

TODO

---

**Engineering**

- face_provider [GIT:private, [lionrocket-inc](https://github.com/lionrocket-inc/)], 2022.06 \
: *All-in-one Face generation API*

얼굴 인식, 검색, 합성, 분류, 추천 목적 통합 서비스 지원 프레임워크 \
Skills: Python, PyTorch, dlib, opencv, FAISS \
R&R: 1인 개발

통합 얼굴 이미지 지원 프레임워크입니다. 이미지 내 얼굴 탐지를 시작으로 정렬, 인식, 분류, 벡터 데이터베이스에서의 검색과 추천을 지원합니다.

얼굴 탐지와 인식 과정에는 입력 이미지의 회전량에 따라 인식 성능이 떨어지는 문제가 있었고, 이를 보정하기 위해 두상의 회전량을 추정하여 이미지를 정면으로 정렬하거나, 인식이 불가능한 이미지를 사전에 고지할 수 있게 구성하였습니다.

이후 검색과 분류, 추천 과정이 실시간으로 이뤄져야 한다는 기획팀의 요청이 있었고, 벡터 검색 과정은 MetaAI의 벡터 검색 시스템 [FAISS](https://github.com/facebookresearch/faiss)를 활용하여 최적화를 진행하였습니다. 초기 분류 모델은 [dlib](http://dlib.net/)의 얼굴 랜드마크를 기반으로 작동하였으나, [dlib](http://dlib.net/)은 실시간 구성이 어렵다는 문제가 있었고, 추후 [Mediapipe](https://google.github.io/mediapipe/) 교체를 고려하고 있습니다.

---

- CULICULI [GIT:private, [lionrocket-inc](https://github.com/lionrocket-inc/)], 2020.07.10 \
: *CUDA Lib for LionRocket*

C++ CUDA Native를 활용하여 딥러닝 추론 속도를 10배 가량 가속화한 프레임워크 \
Skills: C++, CUDA, Python, PyBind \
R&R: 1인 개발

음성 합성 파이프라인의 추론 가속화를 위해 C++ CUDA Native를 활용하여 10배가량 합성 시간을 단축시킨 프로젝트입니다. C++과 CUDA를 통해 기본적인 Tensor 객체와 BLAS(Basic Linear Algebra Subroutines)를 구성하고, 합성 속도를 최적화한 후, [PyBind](https://pybind11.readthedocs.io/en/stable/)를 통해 python 인터페이스를 제공하였습니다.

당시 TTS 모델에는 음성의 길이에 합성 시간이 비례하는 문제가 있었고, 단위 시간을 줄여 거의 실시간에 가까운 합성 속도를 구성할 수 있어야 했습니다. 이를 위해 C++로 BLOB-Shape Tuple 형태의 Tensor 객체를 구축하고, 템플릿 프로그래밍을 통해 이를 CUDA Native에서도 활용할 수 있게 두었습니다.

BLAS 구현과 POC 이후 병목이 메모리 할당에 있음을 확인하여, 메모리 풀과 CUDA API를 활용하지 않는 자체적인 메모리 할당 방식을 구성, 대략 5~7배의 속도 향상을 확인할 수 있었습니다.

이렇게 만들어진 프레임워크를 팀에서 활용하고자 했고, LR_TTS에서 학습된 체크포인트를 파이썬 인터페이스로 실행 가능하도록 [PyBind](https://pybind11.readthedocs.io/en/stable/)를 활용하였습니다.

---

- LR_TTS [GIT:private, [lionrocket-inc](https://github.com/lionrocket-inc/)], 2019.09 \
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