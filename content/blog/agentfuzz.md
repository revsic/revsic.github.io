---
title: "[WIP] Research: AgentFuzz, Agentic Fuzzing Harness Generation with LLM"
date: 2024-11-10T17:33:30+09:00
draft: false

# post thumb
image: "images/post/agentfuzz/head.png"

# meta description
description: "AgentFuzz: Agentic Fuzzing Harness Generation with LLM"

# taxonomies
categories:
  - "Software Testing"
tags:
  - "Software Testing"
  - "Fuzzing"
  - "LibFuzzer"
  - "Large Language Model"
  - "LLM"
  - "Harness"

# post type
type: "post"
---

아래 글은 2024년 3월부터 11월까지 수행한 학부 졸업 프로젝트에 관한 수기이다.

- Research about the Automatic Fuzzing Harness Generation
- Keyword: Software Testing, Fuzzing, Harness Generation, Large Language Model, LLM

**Introduction**

소프트웨어 테스팅에서 Fuzzing은 소프트웨어에 무작위 입력을 대입하여 결함의 발생 여부를 관찰하는 방법론을 일컫는다. Fuzzer는 무작위 입력을 생성하여 소프트웨어에 대입하고, 결함의 관찰을 대행하는 도구를 지칭한다. 경우에 따라 완전히 무작위한 값을 생성하기도 하고, 사용자에 의해 주어진 입력을 변조하여 활용하기도 한다.

일반적으로 무작위 값을 단순히 소프트웨어에 입력할 경우, 대다수는 인터페이스에 해당하는 영역에서 입력값 검증 등 조건 분기를 통과하지 못하고 조기에 종료된다. 이에 내부 구현에 해당하는 영역은 대개 실행되지 않고, 테스트의 범위는 좁아지게 된다. 테스트의 범위를 효과적으로 확장하기 위해서는 다양한 조건 분기를 관측 및 선택하여, 각 경로를 exhaustive하게 탐색하는 것이 유리하다.

분기의 탐색을 효과적으로 수행하기 위해 Fuzzer는 Coverage를 활용하기도 한다. 

소프트웨어 테스팅에서는 소프트웨어의 동작 과정에서 어떤 코드 블럭/분기/함수 등이 몇 번 실행되었는지 기록하여, 이를 "Coverage"라는 이름으로 관리한다. 기존까지는 실행되지 않았던 새로운 코드 블럭/분기/함수가 실행된 경우, 우리는 Coverage가 증가하였다고 표현한다.

몇몇 Fuzzer는 Coverage를 증가시킨 입력에 높은 우선순위를 두고, 우선순위에 따라 과거의 입력을 선택하여 무작위로 변조-소프트웨어에 대입하는 일련의 작업을 반복한다. 이 경우 전략 없이 단순 무작위 입력값을 생성하는 Fuzzer에 비해 높은 확률로 Coverage가 증가하는 방향의 입력이 성생되길 기대할 수 있다.

이렇게 무작위로 값을 변조하는 과정을 Mutation이라 하고, 과거의 입력을 Seed Corpora(복수형 Seed corpus)라 하자. 또한 Coverage 기반의 Mutation 전략을 가지는 Fuzzer를 아래 수기에서는 Greybox Fuzzer라 표현하겠다. 

---

Greybox Fuzzer 역시 한계를 가진다. Coverage 기반의 Mutation 전략을 통해 상대적으로 테스트 범위를 확장할 수 있었지만, 충분히 복잡한 소프트웨어의 분기 구조 속에서 무작위 입력만으로 도달할 수 있는 코드에는 한계가 있다. 또한, 그래픽 인터페이스를 내세운 소프트웨어의 경우, 생성된 입력을 적절한 인터페이스에 전달할 별도의 장치도 필요하다.

이러한 상황 속 더욱 효과적인 테스트를 위해 등장한 것이 Fuzzer Harness이다. Harness는 무작위 입력을 테스트 대상에 전달하는 별도의 소프트웨어로, 무작위 값을 인자로 특정 함수를 호출하거나, 네트워크 혹은 GUI 이벤트를 모방하여 무작위 입력을 인터페이스에 연결한다. Fuzzer는 소프트웨어를 직접 실행하는 대신 Harness를 실행하고, Harness는 Fuzzer가 전달한 무작위 입력을 소프트웨어 전달하게 된다.

이를 통해 테스트 대상의 특정 구현체를 인터페이스의 제약에서 벗어나 직접 Fuzzing 할 수 있게 된다.

다만, 이러한 Harness를 작성하기 위해서는 테스트 대상에 관한 이해가 선행되어야 하며, 테스트 대상에 새 기능이 추가 되거나 수정될 경우 Harness 역시 수정되어야 할 수 있다.

OSS-Fuzz-Gen, PromptFuzz 등 프로젝트는 이에 대응하기 위해 LLM을 활용하여 Harness를 생성하고, 이를 토대로 Fuzzing을 수행한다. Harness 작성 시간을 단축하고, Fuzzing의 진행 경과에 따라 동적으로 테스트가 부족한 영역에 관한 Harness를 보강하여 전반적인 테스트 커버리지를 높여간다.

**OSS-Fuzz-Gen, PromptFuzz**

OSS-Fuzz[[google/oss-fuzz](https://github.com/google/oss-fuzz)]는 구글에서 운영하는 오픈소스 Fuzzing 프로젝트이다. 오픈소스 제공자가 빌드 스크립트와 Fuzzer를 제공하면 구글이 ClusterFuzz[[google/cluster-fuzz](https://github.com/google/clusterfuzz)]를 통해 GCP 위에서 분산 Fuzzing을 구동-결과를 통고해 주는 방식으로 작동한다.

{{< figure src="/images/post/agentfuzz/ossfuzz.png" width="100%" caption="Figure 1. google/oss-fuzz#Overview" >}}

**Problems**

**Approaches**

**Conclusion**

**Future works**
