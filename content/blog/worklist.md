---
title: "Work list"
date: 2020-03-14T01:21:12+09:00
draft: false

# post thumb
image: "images/post/post1_1.jpg"

# meta description
description: "work lists"

# taxonomies
categories: 
  - "Portfolio"
tags:
  - "Portfolio"

# post type
type: "post"
---


**Skills**
1. Languages \
: Python \
: C++ [[git+revsic/cpp-obfuscator](https://github.com/revsic/cpp-obfuscator), [git+utilForever/RosettaStone](utilForever/RosettaStone)]

2. ML Framework \
: Tensorflow [[git+revsic/tf-glow-tts](https://github.com/revsic/tf-glow-tts), [git+revsic/tf-diffwave](https://github.com/revsic/tf-diffwave)] \
: PyTorch [[git+revsic/torch-nansypp](https://github.com/revsic/torch-nansypp), [git+revsic/torch-flow-models](https://github.com/revsic/torch-flow-models)] \
: Jax/Flax [[git+revsic/jax-variational-diffwave](https://github.com/revsic/jax-variational-diffwave)]

3. Windows Internal [[git+revsic/cpp-veh-dbi](https://github.com/revsic/cpp-veh-dbi)]

4. Fuzzing [[git+revsic/agent-fuzz](https://github.com/revsic/agent-fuzz), [git+theori-io/aixcc-afc-archive](https://github.com/revsic/aixcc-afc-archive)]

**Opensource Contributions**

- sgl-project/sglang [[GIT, PR#411](https://github.com/sgl-project/sglang/pull/411)], 2024.05. \
Cohere Command-R chat template supports.

- linfeng93/BiTA [[GIT, PR#4](https://github.com/linfeng93/BiTA/pull/4)], 2024.02. \
Hard-coded path removal.

- SqueezeBits/QUICK [[GIT, PR#3](https://github.com/SqueezeBits/QUICK/pull/3)], 2024.02. \
Exclude router projection layers from QUICK quantization.

- microsoft/TransformerCompression [[GIT, ISSUE#81](https://github.com/microsoft/TransformerCompression/issues/81)], 2024.01. \
Mixtral 8x7B SliceGPT post-train pruning supports.

- casper-hansen/AutoAWQ [[GIT, PR#251](https://github.com/casper-hansen/AutoAWQ/pull/251)], 2023.12. \
Mixtral 8x7B Activation-aware Quantization supports.

**Projects - Fuzzing**

1. BranchFlipper: Agentic Fuzz Blocker Resolution [[GIT](https://github.com/revsic/aixcc-afc-archive), [whitepaper](/pdf/Branch_flipper.pdf)], 2025. \
: *Unlocking Fuzz Blockers with Coverage-Grounded LLMs, part of DARPA AIxCC (Team Theori)*

2. AgentFuzz: Agentic Fuzz Harness Generation [[GIT](https://github.com/revsic/agent-fuzz), [blog](/blog/agentfuzz)], 2024. \
: *LLM Agent-based fuzz-driver generation, inspired by PromptFuzz[[arXiv:2312.17677](https://arxiv.org/abs/2312.17677)]*

**Projects - Machine Learning**

1. AlphaZero Connect6 [[GIT](https://github.com/revsic/AlphaZero-Connect6)], 2018. \
: *AlphaZero training framework for game Connect6 written in Rust with C++, Python interface.*

2. Behavior based Malware Detection Using Branch Data [[GIT](https://github.com/revsic/tf-branch-malware)], 2017. \
: *Classify malware from benign software using branch data via LSTM based on Tensorflow*

**Projects - Windows Internal**

1. cpp-veh-dbi [[GIT](https://github.com/revsic/cpp-veh-dbi)], 2019. \
: *C++ implementation of vectored exception handler based simple dynamic binary instrumentation tools.*

2. Branch Tracer [[GIT](https://github.com/revsic/BranchTracer)], 2019. \
: *C++ implementation of dll-based windows debugger for tracking branching instruction via vectored exception handler.*

3. Code-Injector [[GIT](https://github.com/revsic/CodeInjection)], 2018. \
: *C++ implementation of several code injection techniques like dll injection, queue user apc.*

4. AntiDebugging [[GIT](https://github.com/revsic/AntiDebugging)], 2017. \
: *C++ implementation for defending windows debugger from attaching the target process.*

**Projects**

1. cpp-concurrency [[GIT](https://github.com/revsic/cpp-concurrency)], 2019. \
: *C++ implementation of golang-style concurrency supports, thread pool, channel, wait-group*

2. cpp-obfuscator [[GIT](https://github.com/revsic/cpp-obfuscator)], 2019. \
: *C++ implementation of compile time string and routine obfuscator.*

3. RosettaStone [[GIT](https://github.com/utilForever/RosettaStone)], 2018. \
: *C++ implementation of game 'Hearthstone' as training environment and A.I. for future work.*

4. PacketInjector [[GIT](https://github.com/revsic/PacketInjector)], 2016. \
: *C++ implementation of simple packet detector and injector.*

5. ELF Code Virtualization, 2015. \
: *ELF (Executable Linkable Format) Virtualized Code Protection*

**Paper implementations**

- torch-flow-models [[GIT](https://github.com/revsic/torch-flow-models)], 2025.02. \
: *PyTorch implementations of various generative models, +17*

- torch-nansy++ [[GIT](https://github.com/revsic/torch-nansypp)], 2022.12. \
: *NANSY++: Unified Voice Synthesis with Neural Analysis and Synthesis, openreview, 2022.*

- torch-whisper-guided-vc [[GIT](https://github.com/revsic/torch-whisper-guided-vc)], 2022.12. \
: *Torch implementation of Whisper-guided DDPM based Voice Conversion*

- torch-nansy [[GIT](https://github.com/revsic/torch-nansy)], 2022.09. \
: *Neural Analysis and Synthesis: Reconstructing Speech from Self-Supervised Representations, Choi et al., 2021.*

- torch-retriever-vc [[GIT](https://github.com/revsic/torch-retriever-vc)], 2022.04. \
: *Retriever: Learning Content-Style Representation as a Token-Level Bipartite Graph, Yin et al., 2022.*

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

**School Works**
1. PA-2025-1H [[GIT](https://github.com/revsic/PA-2025-1H)] \
: *Lab notes on "Programming Analysis" in Seoul National University*

2. HYU-ITE2038 [[GIT](https://github.com/revsic/HYU-ITE2038)] \
: *Lab notes on "Database Systems and Applications" in Hanyang University*

3. HYU-CSE4007 [[GIT](https://github.com/revsic/HYU-CSE4007)] \
: *Lab notes on "Artificial Intelligence" in Hanyang University*

4. HYU-ELE3021 [[GIT](https://github.com/revsic/HYU-ELE3021)] \
: *Lab notes on "Operating System" in Hanyang University*

**Papers**
1. Behavior Based Malware Detection Using Branch Data \
: [2017 KIISE Korea Computer Science Conference](https://www.kiise.or.kr/), 2017.

**Presentations**

1. Prompt Engineering Trends \
: Theori, [OpenTRS](https://www.youtube.com/@OpenTRS), 2025.01.16., 2025.02.18.

2. Math for A.I., Generative Models \
: Mapo High School, 2023.11.28.

3. The 2nd AI & Dining. Virtual Human and Generative Models \
: Sangmyung University, 2022.09.22.

4. Deep learning and A.I. \
: Danggok High School, 2022.08.30.

5. 2022 A.I.U. Research generative models in Startup [[Google Drive](https://drive.google.com/file/d/1RT_6LW1cEJfOrVekeV8tQo-j_o63gm2G/view?usp=sharing)] \
: A.I.U. 2022 AI Confrerence, 2022.05.

6. Developing Environment for RL \
: [Nexon Developers Conference 2019](https://ndc.nexon.com/main) as team [RosettaStone](https://github.com/utilForever/RosettaStone), 2019.

7. GP to NP: Gaussian process and Neural Process \
: [A.I.U 1st Open AI Conference](https://festa.io/events/288), 2018.

8. Hearthstone++: Hearthstone simulator with reinforcement learning \
[Deep Learning Camp Jeju](http://jeju.dlcamp.org/2018/), 2018.

9. Behavior based Malware Detection Using Branch Data \
: [CodeGate 2017 Junior](https://www.codegate.org/), 2017.

**Awards**
1. DARPA, AIxCC, 3rd Place; [Team Theori](https://theori-io.github.io/aixcc-public/index.html) \
Defense Advanced Research Projects Agency, AI Cyber Challenge, 3rd Place($1.5M), 2025.08

2. KISA, 2016 Software Contest, \
Application Security Section 2nd Prize (Minister of Interior Award)
2016.09

**Educations**
1. M.S. Department of Computer Science and Engineering \
[Seoul National University](https://www.snu.ac.kr/), [Visual & Geometric Intelligence Lab](https://jaesik.info/lab) (2025.03. ~ )

2. B.S. Major, Department of Computer Science and Engineering \
[Hanyang University](https://www.hanyang.ac.kr/) (2018.03. ~ 2025.02.)

3. Minor, Department of Mathematics \
[Hanyang University](https://www.hanyang.ac.kr/) (2019.03. ~ 2025.02.)

4. Vulnerability Analysis Track \
: [5th KITRI BoB](https://www.kitribob.kr/) (2016.05. ~ 2017.03.)

5. Department of Information and Communication Technology \
: [Sunrin Internet High School](http://sunrint.hs.kr/) (2015.03. ~ 2017.02.)

**Works**

1. AI for Offensive Security, Freelance Researcher \
[Theori](https://theori.io/) (2025.02. ~ 2025.08.)

2. AI for Offensive Security, Senior Researcher \
[Theori](https://theori.io/) (2024.10. ~ 2025.02.)

3. Research Team Lead \
[Theori](https://theori.io/) (2023.08. ~ 2024.10.)

4. Video Synthesis, AI Researcher \
[LionRocket](https://lionrocket.ai) (2021.10. ~ 2022.10., 2023.03. ~ 2023.08.)

5. Research Team Lead \
[LionRocket](https://lionrocket.ai) (2021.04. ~ 2023.08.)

6. Speech Synthesis, AI Researcher \
[LionRocket](https://lionrocket.ai) (2019.09. ~ 2021.10., 2022.10. ~ 2023.02.)
