---
layout: post
title: "LLM 추론과 활용: Sampling부터 RAG 검색 파이프라인까지"
description: "Temperature와 top-p가 출력을 바꾸는 원리, Context Window의 한계, 그리고 Embedding·Hybrid Search·Reranking으로 이어지는 실전 RAG 설계를 정리합니다."
date: 2026-07-30 23:00:00 +0900
tags: [LLM, RAG, Embedding, Inference]
category: development
category_label: 개발
reading_time: 12
math: true
---

LLM을 제품에 연결하면 학습 때와는 다른 질문이 생긴다. 같은 모델이 왜 실행할 때마다 다른 답을 내는가? 긴 문서를 전부 넣었는데 왜 중요한 내용을 놓치는가? 최신 사내 문서를 답하게 하려면 Prompting, RAG, Fine-tuning 중 무엇을 선택해야 하는가?

이 글은 모델을 **어떻게 학습시키는가**보다 이미 학습된 모델을 **어떻게 안정적으로 활용하는가**에 초점을 맞춘다.

```text
Sampling 설정
    ↓
Context 구성
    ↓
Retrieval과 Reranking
    ↓
근거 기반 답변 생성과 평가
```

## 1. Sampling: 다음 토큰을 어떻게 고르는가

언어 모델은 다음 단어 하나를 바로 출력하지 않는다. 먼저 어휘의 각 토큰에 점수인 logit을 부여하고, 이를 확률분포로 바꾼다. Sampling 파라미터는 이 분포에서 실제 토큰을 선택하는 방식을 조절한다.

### Temperature

Temperature $T$는 softmax에 들어가는 logit의 크기를 조절한다.

$$
p_i(T)=\frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
$$

- $T<1$이면 높은 점수의 토큰에 확률이 더 집중된다.
- $T>1$이면 분포가 평평해져 낮은 순위의 토큰도 선택될 가능성이 커진다.
- $T$가 0에 가까워지면 가장 높은 점수의 토큰을 고르는 결정적 선택에 가까워진다.

예를 들어 기본 확률이 다음과 같다고 하자.

| 후보 | 기본 확률 | 높은 Temperature의 예 |
| --- | ---: | ---: |
| 사과 | 50% | 38% |
| 바나나 | 30% | 29% |
| 포도 | 15% | 21% |
| 자동차 | 5% | 12% |

Temperature를 높이면 `사과`가 1위라는 순서는 유지될 수 있지만 후보 사이의 격차가 줄어든다. 창작과 아이디어 발산에는 다양성이 도움이 될 수 있다. 반면 코드 생성, 정보 추출, 분류처럼 재현성이 중요한 작업은 낮은 값에서 시작하는 편이 일반적이다.

낮은 Temperature가 사실의 정확성을 보장하는 것은 아니다. 모델이 잘못된 정보를 높은 확률로 믿고 있다면 더 일관되게 틀릴 수도 있다. 정확성은 검색, 도구 사용, 출력 검증과 별도로 관리해야 한다.

### Top-p

Top-p 또는 nucleus sampling은 확률이 높은 순서대로 토큰을 더해 **누적 확률이 $p$ 이상이 되는 최소 후보 집합**만 남긴다.

| 후보 | 확률 | 누적 확률 |
| --- | ---: | ---: |
| A | 40% | 40% |
| B | 30% | 70% |
| C | 15% | 85% |
| D | 10% | 95% |
| E | 5% | 100% |

`top-p = 0.7`이면 A와 B까지가 후보 집합이 된다. `top-p = 0.9`이면 누적 확률이 처음 90% 이상이 되는 D까지 포함된다.

Temperature는 분포의 **모양**을 바꾸고, top-p는 그 분포에서 고려할 후보의 **범위**를 자른다. 둘을 동시에 바꿀 수도 있지만, 원인을 추적하기 어려워지므로 실험 초기에는 하나씩 조절하는 편이 좋다.

## 2. Context Window: 들어간다고 모두 활용되는 것은 아니다

Context Window는 모델이 한 번의 요청에서 처리할 수 있는 토큰 범위다. 여기에는 사용자 입력뿐 아니라 시스템 지시, 대화 기록, 검색 문서, 도구 결과, 모델이 생성할 출력 예산까지 포함될 수 있다.

입력이 한도를 넘었을 때의 동작은 모델 자체보다 API와 애플리케이션 구현에 달려 있다.

- 요청을 거부하고 길이 초과 오류를 낼 수 있다.
- 오래된 대화나 우선순위가 낮은 문서를 애플리케이션이 잘라낼 수 있다.
- 대화를 요약해 더 짧은 형태로 다시 넣을 수 있다.

잘려서 Context Window 밖으로 나간 정보는 현재 추론에서 직접 참고할 수 없다. 따라서 긴 대화에서는 요약 메모리를 만들거나, 필요한 정보를 외부 저장소에서 다시 가져오는 전략이 필요하다.

### Lost in the Middle

Context Window를 크게 만들면 길이 제한은 완화되지만 검색 문제가 사라지는 것은 아니다. 긴 입력에서는 관련 정보의 위치에 따라 모델의 활용 성능이 달라질 수 있다. 특히 중요한 근거가 중간에 있을 때 놓치는 경향을 `lost in the middle`이라고 부른다.

실무에서는 문서 전체를 무작정 넣기보다 다음과 같이 구성한다.

- 질문과 관련된 문단을 우선 선택한다.
- 필요한 경우 앞뒤 문단과 문서 제목·섹션 경로를 함께 넣는다.
- 가장 중요한 지시와 근거가 쉽게 식별되도록 구조화한다.
- 실제 사용 길이와 위치 조건을 바꿔가며 평가한다.

긴 Context와 RAG는 경쟁 관계가 아니다. 긴 Context는 한 번에 더 많은 근거를 담게 하고, RAG는 그 안에 **무엇을 넣을지** 결정한다.

## 3. Prompting, RAG, Fine-tuning 선택 기준

세 방법은 서로 다른 문제를 해결한다.

| 방법 | 주된 역할 | 먼저 고려할 상황 |
| --- | --- | --- |
| Prompting | 요청 시점의 지시와 예시 제공 | 빠른 실험, 일회성 작업, 기준 성능 확인 |
| RAG | 최신 외부 지식을 검색해 Context에 제공 | 사내 문서, 규정, 제품 정보처럼 자주 바뀌는 지식 |
| Fine-tuning | 모델의 반복적인 행동 패턴 조정 | 형식·말투·도메인 작업 습관이 지속해서 어긋날 때 |

새로운 서비스의 아이디어를 검증하는 단계라면 Prompting부터 시작하는 것이 가장 빠르다. 모델이 최신 회사 규정을 모른다면 RAG가 필요하다. 특정 응답 행동이 충분한 예시와 평가 데이터가 있는데도 반복해서 실패한다면 Fine-tuning을 검토할 수 있다.

하나만 선택해야 하는 것도 아니다. 최신 규정을 근거로 정해진 JSON 형식의 답을 만들어야 한다면 다음과 같이 조합할 수 있다.

```text
RAG로 최신 규정 검색
    +
Schema 기반 Structured Output으로 형식 강제
    +
반복적인 행동 실패가 남을 때 Fine-tuning 검토
```

JSON 형식만 자주 어긴다는 이유로 곧바로 Fine-tuning할 필요는 없다. 사용하는 API가 JSON Schema나 constrained decoding을 지원한다면 이를 먼저 적용하는 편이 더 직접적이다.

## 4. RAG의 시작: Chunking, Embedding, Vector DB

RAG(Retrieval-Augmented Generation)는 질문과 관련된 외부 문서를 찾아 모델의 Context에 넣고 답을 생성하는 구조다.

```text
문서 수집 → Chunking → Embedding → Index 저장
                                      ↓
질문 → Query Embedding → Retrieval → Context 구성 → LLM 답변
```

### Chunking

긴 문서를 그대로 하나의 벡터로 만들면 여러 주제가 섞여 핵심 의미가 흐려질 수 있다. 그래서 문서를 검색 가능한 작은 단위인 chunk로 나눈다.

Chunk가 너무 크면 불필요한 내용까지 Context를 차지한다. 너무 작으면 문장의 전제나 표의 헤더처럼 답에 필요한 맥락이 끊긴다. 일정 길이만 겹치는 overlap은 경계에서 정보가 잘리는 문제를 완화하지만, 중복 검색과 저장 비용을 늘릴 수 있다.

좋은 Chunking은 글자 수만 자르는 일이 아니다. 제목, 문단, 코드 블록, 표, 문서 계층 같은 구조를 보존해야 한다.

### Embedding과 Vector DB

Embedding 모델은 질문과 문서를 의미를 나타내는 벡터로 바꾼다. Vector DB는 문서 벡터를 저장하고 질문 벡터와 가까운 항목을 빠르게 찾는다.

> 질문: “환불은 언제 가능한가요?”<br>
> 문서: “구매 후 7일 이내 반품할 수 있습니다.”

표현은 다르지만 의미가 관련 있으므로 두 벡터가 가까운 위치에 놓이도록 학습된다. 검색 시에는 cosine similarity나 dot product 같은 점수로 가까운 Chunk를 찾는다.

$$
\operatorname{cos}(q,d)=\frac{q\cdot d}{\lVert q\rVert\lVert d\rVert}
$$

Vector DB는 의미를 이해해 정답을 보장하는 데이터베이스가 아니다. 선택한 Embedding 모델과 거리 함수, 인덱스 방식, 데이터 분포에 따라 품질이 달라지는 검색 시스템이며 대규모 환경에서는 근사 최근접 탐색을 주로 사용한다.

## 5. Vector Search만으로 부족한 이유

의미가 비슷한 문서를 찾는 Vector Search는 강력하지만 정확한 제품명, 오류 코드, 사람 이름, 버전 번호처럼 문자열 일치가 중요한 질의에는 약할 수 있다.

### Metadata Filtering

벡터 유사도를 계산하기 전에 문서의 속성으로 범위를 제한할 수 있다.

```text
department = "finance"
effective_date <= today
document_status = "active"
```

Vector Search가 **무슨 내용인지** 찾는다면 Metadata Filtering은 **언제, 어디서, 어떤 상태의 문서인지** 걸러낸다. 권한 정보도 메타데이터로 관리할 수 있지만, 검색 필터만 믿지 말고 애플리케이션의 접근 제어와 함께 적용해야 한다.

### Hybrid Search

Hybrid Search는 의미 기반 Vector Search와 단어 기반 Keyword Search를 결합한다.

- Vector Search: 표현이 달라도 의미가 유사한 문서를 찾는다.
- BM25 같은 Keyword Search: 정확한 용어와 희귀 키워드를 잘 찾는다.
- Fusion: 두 결과의 점수나 순위를 결합한다.

질의 유형이 다양한 실제 서비스에서는 한 가지 검색 방식만 고집하는 것보다 안정적인 경우가 많다.

## 6. MMR과 Reranking: 후보를 더 유용하게 고르기

검색 상위 결과가 모두 같은 내용을 반복하면 Context 공간을 낭비한다. MMR(Maximal Marginal Relevance)은 질문과의 관련성을 높게 유지하면서 이미 선택한 문서와의 중복은 줄이는 방향으로 후보를 고른다.

```text
높은 query relevance
        +
낮은 selected-document redundancy
        ↓
관련 있으면서 다양한 근거
```

반면 Reranking은 첫 번째 검색기가 넓고 빠르게 찾은 후보를 더 정교한 모델로 다시 정렬한다.

```text
Retriever: 수천·수백만 문서에서 빠르게 후보를 찾음
Reranker : 소수 후보를 질문과 함께 읽고 정밀하게 순서를 매김
```

Cross-encoder나 LLM 기반 Reranker는 질문과 문서를 함께 처리해 정확도를 높일 수 있지만 지연 시간과 비용이 늘어난다. 따라서 첫 단계에서 후보를 충분히 회수하고, 두 번째 단계에는 관리 가능한 수만 전달하는 구조가 일반적이다.

MMR과 Reranking도 서로 대체하는 기술은 아니다. Reranking으로 관련성을 높인 뒤 MMR로 다양성을 확보하거나, 제품 요구에 맞는 다른 순서로 조합할 수 있다.

## 7. RAG 실패를 Retrieval과 Generation으로 나누기

RAG가 틀렸을 때 “모델이 환각했다”라고만 결론 내리면 원인을 고치기 어렵다. 먼저 실패를 두 단계로 나눠야 한다.

### Retrieval Failure

정답에 필요한 문서를 가져오지 못한 경우다.

- 문서가 인덱스에 없거나 최신 상태가 아니다.
- Chunk가 너무 크거나 작아 근거가 분리됐다.
- 질의와 Embedding 모델의 언어·도메인이 맞지 않는다.
- `top-k`가 너무 작거나 필터가 과도하다.
- 고유명사 질의에 Vector Search만 사용했다.

이 단계는 정답 Chunk가 상위 $k$개 안에 포함되는지 보는 `Recall@k` 같은 검색 지표로 평가할 수 있다.

### Generation Failure

필요한 문서를 가져왔지만 모델이 잘못 답한 경우다.

- 관련 없는 Chunk가 함께 들어와 근거가 흐려졌다.
- 모델이 문서보다 사전 지식을 우선했다.
- 여러 문서의 날짜나 버전을 구분하지 못했다.
- 질문에 답할 근거가 없는데도 답을 생성했다.

이 단계는 답변의 정확성뿐 아니라 인용한 근거가 주장을 실제로 뒷받침하는지, 근거가 없을 때 답변을 보류하는지까지 평가해야 한다.

## 8. 실전 RAG 파이프라인

원문의 흐름을 실제 시스템 설계 순서로 정리하면 다음과 같다.

```text
1. 문서 파싱과 구조 기반 Chunking
2. Chunk와 Metadata 저장
3. Embedding 생성과 Vector Index 구축
4. Query 전처리 또는 재작성
5. Metadata Filtering
6. Vector + Keyword Hybrid Retrieval
7. Reranking과 중복 제거
8. Context 구성과 LLM 생성
9. 출처 표시, 검증, 로그 수집
10. Retrieval과 Generation을 분리해 평가
```

RAG의 품질은 마지막 LLM 하나로 결정되지 않는다. 문서 파싱, Chunking, 검색, Reranking, Context 구성 중 어느 한 단계만 흔들려도 답이 틀릴 수 있다. 그래서 처음부터 복잡한 파이프라인을 만드는 것보다 단순한 기준선을 만든 뒤 실제 실패 사례를 분류하며 한 단계씩 추가하는 편이 좋다.

결론적으로 Sampling은 **어떤 토큰을 선택할지**, Context 설계는 **모델이 무엇을 볼지**, RAG는 **어떤 근거를 가져올지** 결정한다. 이 세 층을 분리해 이해하면 모델의 문제와 시스템의 문제를 구분할 수 있고, 막연한 Fine-tuning 대신 고칠 지점을 정확히 찾을 수 있다.

## 더 읽어보기

- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
