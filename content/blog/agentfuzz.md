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

소프트웨어 테스팅에서 Fuzzing은 소프트웨어에 무작위 입력을 대입하여 결함의 발생 여부를 관찰하는 방법론을 일컫는다. 무작위 입력을 생성하여 소프트웨어에 대입하고, 결함의 관찰을 대행하는 도구를 Fuzzer라 지칭한다. 경우에 따라 완전히 무작위한 값을 생성하기도 하고, 사용자에 의해 주어진 입력을 활용하기도 한다.

일반적으로 무작위 값을 단순히 소프트웨어에 입력할 경우, 대다수는 인터페이스에 해당하는 영역에서 입력값 검증 등 조건 분기를 통과하지 못하고 조기에 종료된다. 이에 내부 구현에 해당하는 영역은 대개 실행되지 않고, 테스트의 범위는 좁아지게 된다. 테스트의 범위를 효과적으로 확장하기 위해서는 다양한 조건 분기를 관측 및 선택하여, 각 경로를 exhaustive하게 탐색하는 것이 유리하다.

분기의 탐색을 효과적으로 수행하기 위해 Fuzzer는 Coverage를 활용하기도 한다. 

소프트웨어 테스팅에서는 소프트웨어의 동작 과정에서 어떤 코드 블럭/분기/함수 등이 몇 번 실행되었는지 기록하여, 이를 "Coverage"라는 이름으로 관리한다. 기존까지는 실행되지 않았던 새로운 코드 블럭/분기/함수가 실행된 경우, 우리는 Coverage가 증가하였다고 표현한다.

몇몇 Fuzzer는 Coverage를 증가시킨 입력에 높은 우선순위를 두고, 우선순위에 따라 과거의 입력을 선택하여 무작위로 변조-소프트웨어에 대입하는 일련의 작업을 반복한다. 이 경우 전략 없이 단순 무작위 입력값을 생성하는 Fuzzer에 비해 높은 확률로 Coverage가 증가하는 방향의 입력이 성생되길 기대할 수 있다.

이렇게 무작위로 값을 변조하는 과정을 Mutation이라 하고, 과거의 입력을 Seed Corpora(복수형 Seed corpus)라 하자. 또한 Coverage 기반의 Mutation 전략을 가지는 Fuzzer를 아래 수기에서는 Greybox Fuzzer라 표현하겠다. 

---

Greybox Fuzzer 역시 한계를 가진다. Coverage 기반의 Mutation 전략을 통해 상대적으로 테스트 범위를 확장할 수 있었지만, 충분히 복잡한 소프트웨어의 분기 구조 속에서 무작위 입력만으로 도달할 수 있는 코드에는 한계가 있다. 또한, 그래픽 인터페이스를 내세운 소프트웨어의 경우, 생성된 입력을 적절한 인터페이스에 전달할 별도의 장치도 필요하다.

이러한 상황 속 더욱 효과적인 테스트를 위해 등장한 것이 Fuzzer Harness이다. Harness는 무작위 입력을 테스트 대상에 전달하는 별도의 소프트웨어로, 무작위 값을 인자로 특정 함수를 호출하거나, 네트워크 혹은 GUI 이벤트를 모방하여 무작위 입력을 인터페이스에 연결한다. Fuzzer는 Harness를 실행하고, Harness는 Fuzzer가 생성한 무작위 입력을 소프트웨어에 전달한다.

이를 통해 인터페이스의 제약에서 벗어나 테스트 대상의 특정 구현체를 직접 Fuzzing 할 수 있다.

다만, 이러한 Harness를 작성하기 위해서는 테스트 대상에 관한 이해가 선행되어야 하며, 테스트 대상에 새 기능이 추가 되거나 수정될 경우 Harness 역시 수정되어야 할 수 있다.

OSS-Fuzz-Gen, PromptFuzz 등 프로젝트는 이에 대응하기 위해 LLM을 활용하여 Harness를 생성, Fuzzing을 수행한다. Harness 작성 시간을 단축하고, Fuzzing의 진행 경과에 따라 동적으로 테스트가 부족한 영역에 Harness를 보강하여 전반적인 테스트 커버리지를 높여간다.

**Relative Works: OSS-Fuzz-Gen**

OSS-Fuzz[[google/oss-fuzz](https://github.com/google/oss-fuzz)]는 구글에서 운영하는 오픈소스 Fuzzing 프로젝트이다. 오픈소스 제공자가 빌드 스크립트와 Fuzzer를 제공하면 구글이 ClusterFuzz[[google/cluster-fuzz](https://github.com/google/clusterfuzz)]를 통해 Google Cloud Platform(이하 GCP) 위에서 분산 Fuzzing을 구동-결과를 통고해 주는 방식으로 작동한다.

{{< figure src="/images/post/agentfuzz/ossfuzz.png" width="100%" caption="Figure 1. google/oss-fuzz#Overview" >}}

일부 오픈소스 프로젝트에 대해 OSS-Fuzz는 LLM 기반으로 Harness를 생성-테스트하는 일련의 파이프라인을 제공한다; OSS-Fuzz-Gen[[google/oss-fuzz-gen](https://github.com/google/oss-fuzz-gen)].

OSS-Fuzz는 Fuzz-introspector[[ossf/fuzz-introspector](https://github.com/ossf/fuzz-introspector)]를 통해 ClusterFuzz의 실행 결과로부터 어떤 함수가 얼마나 호출되었고, 어떤 분기에 의해 후속 함수의 호출이 불발되었는지 분석-전달한다(i.e. fuzz-blocker, Figure 2.). OSS-Fuzz-Gen은 테스트가 미진한(호출되지 않았거나, 테스트 범위에 포함되지 않은) 함수를 fuzz-introspector의 보고서로부터 발췌하여 LLM에게 해당 함수의 Harness 생성을 요청한다.

{{< figure src="/images/post/agentfuzz/overlay-1.png" width="80%" caption="Figure 2. Reachability tree with coverage overlay (ref:git+ossf/fuzz-introspector)" >}}

e.g. Prompt (from:[oss-fuzz-llm-targets-public](https://storage.googleapis.com/oss-fuzz-llm-targets-public/index.html)):
```md {style=github}
You are a security testing engineer who wants to write a C++ program to execute all lines in a given function by defining and initialising its parameters in a suitable way before fuzzing the function through `LLVMFuzzerTestOneInput`.

Carefully study the function signature and its parameters, then follow the example problems and solutions to answer the final problem. YOU MUST call the function to fuzz in the solution.

Try as many variations of these inputs as possible. Do not use a random number generator such as `rand()`.

All variables used MUST be declared and initialized. Carefully make sure that the variable and argument types in your code match and compiles successfully. Add type casts to make types match.

You MUST call `CJSON_PUBLIC(cJSON_bool) cJSON_Compare(const cJSON *const a, const cJSON *const b, const cJSON_bool case_sensitive)` in your solution!

*SKIPPED*

Problem:
``
CJSON_PUBLIC(cJSON_bool) cJSON_Compare(const cJSON *const a, const cJSON *const b, const cJSON_bool case_sensitive)
``
```

Generated:
```c {style=github}
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "../cJSON.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size); /* required by C89 */

int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    cJSON *a = cJSON_Parse((const char*)data);
    cJSON *b = cJSON_Parse((const char*)data);
    if(a == NULL || b == NULL) return 0;

    cJSON_Compare(a, b, 0);
    cJSON_Compare(a, b, 1);

    cJSON_Delete(a);
    cJSON_Delete(b);

    return 0;
}

#ifdef __cplusplus
}
#endif
```

LLM은 함수의 Spec을 입력으로 Harness를 작성한다(주로 OpenAI GPT, Google Gemini). 단번에 Syntax Error가 없는 Harness를 생성하기는 어려우므로, OSS-Fuzz-Gen은 컴파일 에러 메시지를 LLM에게 전달하여 오류 수정을 요구한다.

```md {style=github}
Given the following C program and its build error message, fix the code without affecting its functionality. First explain the reason, then output the whole fixed code.
If a function is missing, fix it by including the related libraries.

Code:
``
CJSON_PUBLIC(cJSON_bool) cJSON_Compare(const cJSON *const a, const cJSON *const b, const cJSON_bool case_sensitive)
``

Solution:
``
#include <stdlib.h>
/* *SKIPPED* */
``

Build error message:
/src/cjson/fuzzing/cjson_read_fuzzer.c:1:1: error: unknown type name 'CJSON_PUBLIC'
CJSON_PUBLIC(cJSON_bool) cJSON_Compare(const cJSON *const a, const cJSON *const b, const cJSON_bool case_sensitive)
^
/src/cjson/fuzzing/cjson_read_fuzzer.c:1:25: error: expected ';' after top level declarator
CJSON_PUBLIC(cJSON_bool) cJSON_Compare(const cJSON *const a, const cJSON *const b, const cJSON_bool case_sensitive)
                        ^
                        ;

Fixed code:
```

최대 3~5회까지 수정을 반복하여 Syntax Error를 수정하고, 컴파일에 성공한 경우 최초 시동을 통해 Harness가 Fuzzing과 무관히 Crash를 내는지 확인한다. Fuzzing 전부터 Crash가 발생한다면, 생성된 Harness를 활용하여 Fuzzing을 수행하는 것이 무의미할 것이다.

FYI. 끝내 Syntax Error에 실패할 경우 해당 Harness는 포기하고, LLM에게 새 Harness 합성을 요구한다.

정상 작동한 Harness는 ClusterFuzz로 전달되고, Fuzzing이 이뤄진다.

OSS-Fuzz-Gen은 LLM을 활용하여 tinyxml2 등 프로젝트에서 Test Coverage를 30%까지 추가 획득하였다고 이야기한다[[googleblog](https://security.googleblog.com/2023/08/ai-powered-fuzzing-breaking-bug-hunting.html)].

**Relative Works: PromptFuzz**

OSS-Fuzz-Gen은 LLM을 기반으로 가용한 Harness를 생성할 수 있다는 점을 보였다. 하지만, 대개 함수 개개에 대한 Harness를 작성하기에, API 간의 유기 관계를 테스트하는 것에는 한계가 있다. 특히나 Internal State를 공유하고, 이에 따라 조건 분기를 취하는 라이브러리의 경우, 어떻게 API를 조합하느냐에 따라 trigging할 수 있는 코드 블럭의 부류가 달라질 수 있다. 

PromptFuzz[[arXiv:2312.17677](https://arxiv.org/abs/2312.17677), [git+PromptFuzz/PromptFuzz](https://github.com/PromptFuzz/PromptFuzz)]는 이에 대응하고자 여러 API를 하나의 Harness에서 동시에 호출하는 방식을 취하고, 어떤 API를 선택하는 것이 테스트에 유리한지 새로운 전략을 제시한다.

{{< figure src="/images/post/agentfuzz/workflow.png" width="100%" caption="Figure 3. PromptFuzz/PromptFuzz#workflow" >}}

PromptFuzz는 라이브러리의 헤더 파일로부터 AST 파서를 활용해 함수(API) 및 타입의 선언을 발췌, Gadget이라는 이름으로 관리한다. PromptFuzz는 매 Round마다 이 중 일부를 선택하여 LLM에 Harness 생성을 요구한다.

PromptFuzz는 생성된 Harness의 유효성, Correctness를 검증하기 위한 몇 가지 방법론을 제안하며, 이를 모두 통과한 Harness에 대해서만 Fuzzing을 수행한다.

**PromptFuzz: Harness Mutation**

가장 먼저 고민한 문제는 어떤 API Gadget을 골라 Harness를 만드는가이다. PromptFuzz가 API의 유기 관계를 모델링하기 위해 선택한 방식은 상용 Fuzzer가 Seed Corpus를 Mutation 하는 정책과 동일 선상에 있다.

상용 Fuzzer는 유전 알고리즘을 통해 Coverage가 높은 Seed Corpora를 선택하고, 이를 무작위로 조작하여(random mutation을 가하여) 새로운 입력을 생성한다. Coverage를 측정하여 상위부터 Mutation을 수행하는 일련의 과정을 반복하며, Coverage를 높일 입력을 찾아나가는 것이다.

PromptFuzz는 API Gadget의 순열(이하 API Sequence)을 잘 선택하여 Harness를 구성, 테스트 범위를 확장하길 바란다. 그렇기에 API Sequence를 평가할 지표를 두어 서로 다른 API Sequence 사이에 순서를 정하고, 상위 API Sequence부터 Random Mutation을 수행하여 LLM에게 Harness 생성을 요청한다.

```py {style=github}
## PSEUDO CODE OF PROMPTFUZZ
type APISequence = list[APIGadget]

def round(seed_harnesses: list[Harness]):
    # selection
    selected: Harness = weighted_sample(
        seed_harnesses,
        weight_fn=quality_measure,
    )
    # mutation
    new_api_sequence: APISequence = mutation(selected)
    # generate to harness
    harness: Harness = LLM(
        SYSTEM_PROMPT,
        f"Generate a fuzzer harness containing the given APIs: {new_api_sequence}",
    )
    # validation
    if not is_valid(harness):
        raise ValidationFailureError()
    # run the fuzzer
    result = run_fuzzer(harness)
    # append to seeds
    seed_harnesses.append(harness)
    return result


seed_harnesses = []
# run the PromptFuzz
for _ in range(max_round):
    logger.log(round(seed_harnesses))
```

PromptFuzz는 Harness 역시 Mutation의 대상으로 바라보아 전략적으로 테스트 범위 확장을 의도한다.

Greybox Fuzzer가 Coverage를 Seed Corpus 평가의 지표로 두었다면, PromptFuzz는 API Sequence에 대해 Quality라는 지표를 제안한다.

**Measure**

Quality는 Density와 #Unique Branches의 곱으로 표현된다. Harness Mutation의 목표는 Coverage 확보이다. Mutated Harness를 통해 Coverage(혹은 #Unique Branches)가 얼마나 확보되었는지 파악하는 것은 자명한 일이다. 여기서 중요한 것은 Density의 역할이다.

FYI. #Unique Branches는 Harness를 단위 시간 동안 Fuzzing 하였을 때, Harness에 의해 실행된 대상 프로젝트 내 분기의 수이다. 대상 프로젝트의 Coverage는 #Unique Branches를 프로젝트 내 전체 분기의 수로 나눈 것과 같다.

$$\mathrm{Quality}(g) = \mathrm{Density}(g) \times (1 + \mathrm{UniqueBranches}(g))$$

PromptFuzz는 Harness 내 API의 유기 관계를 적극 활용하여 Coverage를 높이고자 한다. API의 유기 관계에 대한 평가 지표가 제안되어야 하고, 해당 지표가 Coverage 확보에 기여함을 보인다면 명쾌할 것이다.

PromptFuzz가 이를 위해 제안한 지표가 Density이다. API의 유기 관계는 앞선 API의 호출이 후속 API의 실행 흐름에 얼마나 영향을 미치는지로 표현된다. 한 API의 호출이 다른 API의 실행 흐름에 영향을 주기 위해서는 (1) 앞선 호출이 프로젝트의 State를 변화시켜, 후속 실행 흐름에 간접적 영향을 주거나 (2) 앞선 호출의 결과값이 후속 API의 인자로 전달되어 직접적 영향을 주는 2가지 경우로 나뉠 것이다.

Density는 이중 후자의 경우에 집중한다. Harness 내에 존재하는 API를 Node로 표현하고, Taint Analysis를 통해 Harness의 실행 흐름 중 한 API의 반환값이 다른 API의 인자로 전달되는 경우를 Directed Edge로 하여 API Call Depedency Graph를 그린다.

만약 아래의 Harness가 있다면, 다음의 CDG를 예상해 볼 수 있다.

```cpp {style=github}
vpx_codec_dec_cfg_t dec_cfg = {0};
...
// Initialize the decoder
vpx_codec_ctx_t decoder;
vpx_codec_iface_t *decoder_iface = vpx_codec_vp8_dx();
vpx_codec_err_t decoder_init_res = vpx_codec_dec_init_ver(
    &decoder, decoder_iface, &dec_cfg, 0, VPX_DECODER_ABI_VERSION);
if (decoder_init_res != VPX_CODEC_OK) {
    return 0;
}
// Process the input data
vpx_codec_err_t decode_res = vpx_codec_decode(&decoder, data, size, NULL, 0);
if (decode_res != VPX_CODEC_OK) {
    vpx_codec_destroy(&decoder);
    return 0;
}
// Get the decoded frame
vpx_image_t *img = NULL;
vpx_codec_iter_t iter = NULL;
while ((img = vpx_codec_get_frame(&decoder, &iter)) != NULL) {
    // Process the frame
    vpx_img_flip(img);
    ...
}
// Cleanup
vpx_codec_destroy(&decoder);
return 0;
```

{{< figure src="/images/post/agentfuzz/cdg.png" width="50%" caption="Figure 4. Call Dependency Graph" >}}

FYI. 위는 예시이며, 실제 구현과는 다를 수 있다.

Graph는 SCC로 분해가능하고, 각 Component의 Cardinality(집합 내 원소의 수) 중 가장 큰 값을 Density라 명명한다.

FYI. SCC(Strongly Connected Component): 노드의 집합, (1) 집합 내 어떤 임의의 두 노드를 선택하여도 이를 잇는 경로가 존재하고-Strongly Connected, (2) Graph 내 어떤 두 노드가 Strongly Connected이면 둘은 같은 SCC에 속함-Component. (Strongly Connected Nodes의 집합 중 가장 크기가 큰 집합.)

FYI. Graph는 SCC로 Partition 가능하다. (i.e. Graph는 SCC의 집합으로 표현 가능하고, Graph 내 모든 SCC는 mutually disjoint이다.)

위 CDG는 기재된 모든 함수 사이에 서로를 잇는 Edge가 존재하므로 Graph 전체가 하나의 SCC이며, Density는 SCC 내 노드의 개수인 6이다. 

Density는 Harness 내 직접적 영향을 주고 받는 API의 군집 중 가장 큰 군집의 크기를 의미한다. Density가 크다는 것은 Harness 내의 API 유기 관계에 부피감이 있음을 의미한다. (1) 이는 너비를 의미할 수도 있고-여러 API의 독립적 실행 결과가 하나의 API에 영향을 가함, (2) 깊이를 의미할 수도 있으며-API의 호출이 순차적으로 영향을 가함, (3) 이 둘 모두를 의미할 수도 있다. 

Density는 Taint Analysis의 범위에 따라 간접 영향에 관하여는 모델링하지 못할 수도 있고, 그 부피감이 어떤 형태의 Call Dependency를 가지는지 묘사하지 못하기도 한다. 

결국 Quality는 (1) Coverage가 높을수록 (2) API의 유기 관계에 부피감이 있을수록 좋은 Harness라 정의하고 있다. 

**Mutation**

Quality에 따라 Harness가 선택되고 나면 PromptFuzz는 Mutation을 수행한다. Byte string을 직접 조작하는 Corpus Mutation과 달리, Harness Mutation은 API Sequence 수준에서 Mutation을 가하고, LLM을 통해 Mutated API Sequence를 새로운 Harness로 생성하는 과정을 거친다.

$$\mathrm{Harness} \mapsto \mathrm{API\ Sequence} \mapsto \mathrm{Mutated} \mapsto \mathrm{New\ Harness}$$

LLM이 Mutated API Sequence를 토대로 Harness를 만들어도, 제시된 API가 모두 Harness에 포함되어 있지는 않다(LLM의 한계). 따라서 Harness에서 사용한 API를 모두 발췌하여 실행 순서에 따라 Topological Sort를 수행, 생성된 Harness의 API Sequence로 정의한다.

API Sequence에는 (1) API Insert, (2) API Remove, (3) Crossover 3가지 방식의 Mutation 중 하나를 무작위 선택하여 가하게 된다.

API Insert와 Remove는 주어진 API Sequence의 임의 지점에 새로운 API Gadget을 삽입하거나, 임의 지점의 API Gadget을 제거하는 방식으로 작동한다. Crossover는 또 다른 API Sequence와 임의 지점에서 Sequence 전-후반을 접합하는 방식으로 작동한다.

```py {style=github}
## PSEUDO CODE OF MUTATION
def insert(harness: Harness, gadgets: list[APIGadget]):
    seq: APISequence = extract_apis(harness)
    while True:
        api: APIGadget = weighted_sample(
            gadgets,
            weight_fn=energy_measure,
        )
        if api not in seq:
            break
    seq.insert(random.randint(0, len(seq)), api)
    return seq


def remove(harness: Harness):
    seq: APISequence = extract_apis(hanress)
    # inverse energy order
    api: APIGadget = weighted_sample(
        seq,
        weight_fn=lambda x: 1 / energy_measure(x),
    )
    seq.remove(api)
    return seq


def crossover(harness: Harness, seed_harnesses: list[Harness]):
    seq: APISequence = extract_apis(harness)

    other: Harness = weighted_sample(
        seed_harnesses,
        weight_fn=quality_measure,
    )
    other_seq: APISequence = extract_apis(other)
    i = random.randint(1, len(seq) - 1)
    j = random.randint(1, len(other_seq) - 1)
    return seq[:i] + other_seq[j:]


new_api_sequence: APISequence
match random.randint(0, 2):
    case 0: new_api_sequence = insert(harness, gadgets)
    case 1: new_api_sequence = remove(harness)
    case 2: new_api_sequence = crossover(harness, seed_harnesses)
return new_api_sequence
```

Crossover는 역시 Quality에 따라 Harness를 추가 선발하여 활용한다. 동일 논리라면, Insert와 Remove 역시 추가 혹은 제거 대상으로 삼을 API의 기준이 필요할 것이다. 

프로젝트에 따라 PromptFuzz에서 발췌된 API Gadget은 만여개 단위까지 늘어난다. 이 중에는 실제로도 자주 쓰이는 API로, LLM 역시 단번에 활용처를 이해하고 컴파일까지 성공하는 API가 있는반면, 자주 쓰이지 않아 LLM 역시 컴파일에 실패하거나 빈번히 오사용하는 API도 있다. 

PromptFuzz는 이러한 상황에서 이미 충분히 테스트 되었다 판단된 API의 사용을 줄이고, 테스트 되지 않은 API의 사용 시도를 늘리기 위해 Insert와 Remove의 대상에 Energy라는 기준을 제시한다.

$$\mathrm{Energy}(a) = \frac{1 - \mathrm{Coverage}(a)}{(1 + \mathrm{Seed}(a))^E \times (1 + \mathrm{Prompt}(a))^E}$$

Energy는 각 API에 대한 평가 지표로, Energy가 높을수록 Mutation 후 Harness에 해당 API가 잔존할 확률을 높이고, Energy가 낮을수록 잔존 확률을 낮춘다.

FYI. Coverage(a)는 전체 API $a$ 내부 분기 중 실행된 분기의 비율. Prompt(a)는 mutated api sequence에 API $a$가 포함된 횟수(LLM에게 전달된 횟수). Seed(a)는 실제 API $a$를 포함하고 있는 Seed Harnesses의 수(API가 LLM에게 합성 요청되어도 실제 Harness에 포함되지 않을 수 있고, 포함되더라도 Validation 단계를 통과하지 못해 Seed Harnesses에 포함되지 않을 수 있음.)

FYI. E는 하이퍼파라미터, git+PromptFuzz/PromptFuzz는 1로 가정.

API가 충분히 테스트 되었다 판단될수록(i.e. Coverage가 100%에 가까워질수록) API는 Mutated Harness에 포함될 가능성이 줄어든다. 이는 자명하다. 

어떤 API는 LLM에게 많이 합성 요청되었지만, 컴파일 오류나 오사용으로 인해 Fuzzing 대상이 되지 못할 수 있다. 이 경우는 LLM의 성능상 한계라 이해하고, Energy를 통해 해당 API 역시 Mutated Harness에 포함될 가능성을 낮춘다.

결국 PromptFuzz는 Quality와 Energy를 통해 API가 고루 테스트 될 수 있도록 하고, 지표 기반 Mutation으로 좋은 Harness를 찾아나간다.

**PromptFuzz: Harness Validation**

생성된 Harness는 유효성, Correctness를 검증받게 된다. Syntax Error를 포함하여 컴파일이 불가능하거나, API의 오사용으로 인해 새로이 탐색 가능한 분기가 없다면 굳이 이를 구동할 이유가 없을 것이다. PromptFuzz는 효과적인 Fuzzing을 위해 몇 가지 검증 기준을 제안한다.

가장 간단히는 컴파일이 가능해야 한다. LLM의 응답으로부터 \```의 코드 블록이 존재한다면, 블록 내에서 코드를 발췌-컴파일을 시도한다. Syntax Error가 발생할 경우 LLM에게 오류 수정을 요구하는 OSS-Fuzz-Gen과 달리 PromptFuzz는 곧장 생성된 Harness를 폐기하고, 새로 생성을 시도한다.

컴파일에 성공했다면, 최대 10분간 Fuzzer를 구동한다. 1분 단위로 현재 Fuzzer의 Coverage를 측정하여, Coverage가 증가할 경우 지속-유지될 경우 구동을 중지한다. 이후 기존까지 실행되었던 Seed Harnesses의 Fuzzing 결과와 비교하여 새로운 분기가 발견되었는지 검사한다. 만약 분기가 발견되지 않았다면, 현재 검토 중인 Harness는 Coverage 확보에 기여하기 어렵다 판단하여 폐기한다.

만약 컴파일에도 성공하고, 새로운 분기도 확인하였다면 *Critical Path*의 마지막 검증을 거친다. 

**Critical Path**

Critical Path는 Harness 내 여러 Control Flow 중 가장 많은 API를 호출할 수 있는 흐름을 의미한다. 예를 들면, Figure 4.의 예시에서 Critical Path는 다음 6개 API를 포함하고 있다.

vpx_codec_vp8_dx > vpx_codec_dec_init_ver > vpx_codec_decode > vpx_codec_get_frame > vpx_img_flip > vpx_codec_destroy

PromtpFuzz는 최대 10분간의 Harness 구동 중 Critical Path 내의 모든 API가 Hit 되었는지 검사한다. 

만약 생성된 Harness가 API를 오사용하였다면, 테스트를 시도하는 API 중 일부는 실행조차 되지 않고 중도에 종료될 것이다. 반대로 모든 API가 사용되었는지를 테스트한다면, 주류 흐름 외의 에러 핸들링에 사용되는 API까지 강제되는 등 통과가 불가능하거나 비효율적인 평가가 이뤄질 수 있다.

이에 PromptFuzz는 주류 API 흐름의 실행을 보장하고자 Critical Path를 정의하고, 주류 흐름 내의 모든 API가 실행되었는지를 검토한다.

**PromptFuzz: Benchmarks**

TBD; taxonomy of benchmarks

**Problems**

TBD; Syntax errors, Costs, etc.

**Approaches**

TBD; Agentic harness generation, Reusing validation-failed harness

**Conclusion**

**Future works**
