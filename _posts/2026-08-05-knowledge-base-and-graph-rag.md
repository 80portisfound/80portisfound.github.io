---
layout: post
title: "LLM 지식베이스 설계: Hybrid Search부터 Graph RAG와 Provenance까지"
description: "RAG보다 넓은 지식베이스의 생명주기와 Hybrid Search, 지식 그래프, Graph RAG, 출처·삭제 전파를 운영 관점에서 정리합니다."
date: 2026-08-05 16:00:00 +0900
tags: [Knowledge Base, Graph RAG, Hybrid Search, Knowledge Graph]
category: development
category_label: 개발
reading_time: 12
---

RAG를 공부하다 보면 지식베이스를 Vector DB와 거의 같은 의미로 사용하기 쉽다. 하지만 지식베이스는 검색 인덱스 하나보다 훨씬 넓은 시스템이다. 원본 데이터를 수집하고 정제하며, 구조화해 저장하고, 질문에 맞는 근거를 찾아 답변과 연결하고, 변경과 삭제를 계속 반영해야 한다.

```text
데이터 수집 → 파싱·정제 → 구조화 → 저장 → 검색
    → 답변 → 출처 연결 → 업데이트·삭제 → 평가
```

RAG는 이 시스템에 저장된 지식을 LLM이 사용할 수 있게 만드는 방법 중 하나다. 이 글에서는 기존의 RAG 검색 기초를 넘어, 지식베이스가 시간이 지나도 정확하게 유지되려면 어떤 구조가 필요한지 살펴본다.

## 1. RAG와 지식베이스는 같은 것이 아니다

RAG(Retrieval-Augmented Generation)는 질문과 관련된 정보를 검색해 LLM의 Context에 넣고 답변을 생성하는 패턴이다. 지식베이스는 그 검색 대상이 되는 지식을 만들고 관리하는 전체 체계다.

| 구분 | 핵심 질문 |
| --- | --- |
| RAG | 이 질문에 답하기 위해 어떤 근거를 가져올 것인가? |
| 지식베이스 | 어떤 지식을 어떻게 수집·구조화·보관·갱신·삭제할 것인가? |

문서 Chunk와 Embedding을 한 번 만들어 Vector DB에 넣었다고 시스템이 완성되는 것은 아니다. 원본 문서가 수정되거나 폐기됐는데 인덱스가 그대로라면 검색은 오래된 내용을 반환한다. 답변에 출처가 표시돼도 그 출처가 현재 유효한지 판단할 수 없다면 신뢰하기 어렵다.

지식베이스의 품질은 검색 정확도뿐 아니라 **최신성, 추적 가능성, 삭제 완전성, 접근 권한, 평가 가능성**으로 결정된다.

## 2. 지식의 생명주기를 설계하기

지식은 저장된 뒤 끝나는 정적 데이터가 아니다.

```text
수집 → 저장 → 검색 → 평가 → 갱신·삭제 → 다시 평가
```

### 변경 감지

원본 문서의 추가, 수정, 삭제를 알아내야 한다. 파일 수정 시각만 비교할 수도 있지만, 동기화 누락이나 의미 없는 포맷 변경을 구분하기 어렵다. 실무에서는 문서 ID, 버전, content hash, 수집 시각을 함께 관리하는 편이 안전하다.

### 선택적 업데이트

문서 한 줄이 바뀌었다고 전체 지식베이스를 다시 만들 필요는 없다. 변경된 문서를 다시 파싱하고, 영향을 받은 Chunk만 재생성해 Embedding과 검색 인덱스를 갱신할 수 있다.

이때 Chunk 경계가 달라지면 이전 Chunk와 새 Chunk를 단순히 위치 번호로 대응하기 어렵다. 안정적인 문서 ID와 Chunk ID, 원본 범위 정보를 저장해야 변경 범위를 계산할 수 있다.

### 삭제

원본 문서가 폐기되면 연결된 데이터를 함께 제거해야 한다.

- 원문과 Chunk
- Embedding과 검색 인덱스 항목
- 추출된 Entity와 Relation의 근거
- 캐시된 답변과 요약
- 문서에만 존재하던 메타데이터

일부만 남으면 검색에서 더 이상 존재하지 않는 지식이 나타나는 `ghost data`가 생긴다. 삭제는 단일 행을 지우는 작업이 아니라 파생 데이터의 영향 범위를 추적하는 작업이다.

## 3. Hybrid Search: 정확한 이름과 의미를 함께 찾기

Vector Search는 표현이 달라도 의미가 비슷한 문서를 찾는 데 강하다. Keyword Search는 함수명, 오류 코드, 버전, 제품명처럼 정확한 문자열이 중요한 질의에 강하다. Hybrid Search는 두 결과를 결합한다.

```text
Keyword Search ─┐
                ├─ Score 또는 Rank Fusion → Reranking
Vector Search ──┘
```

코드 지식베이스에서 `create_engine`을 찾는 상황을 생각해보자.

- `create_engine`을 정확히 포함한 Chunk는 높은 키워드 점수를 받는다.
- “데이터베이스 연결을 초기화한다”는 설명은 높은 벡터 유사도를 받을 수 있다.
- Python 파일만 찾는다면 언어·파일 형식 메타데이터로 범위를 제한한다.
- 실제 호출 위치가 필요하면 함수 호출 관계 그래프를 탐색한다.

Hybrid Search를 단순히 서로 다른 점수의 평균으로 구현하면 검색기별 점수 범위 때문에 결과가 왜곡될 수 있다. 점수를 정규화하거나 Reciprocal Rank Fusion처럼 순위를 결합한 뒤, Reranker로 최종 후보를 정밀하게 비교할 수 있다.

## 4. 질의 유형에 따라 검색 전략을 바꾸기

모든 질문에 같은 검색 파이프라인을 적용할 필요는 없다.

| 질의 유형 | 예시 | 우선할 검색 방식 |
| --- | --- | --- |
| Exact query | `create_engine`은 어디서 호출되는가? | Keyword + 코드 관계 |
| Semantic query | DB 연결을 초기화하는 코드는? | Vector Search |
| Metadata query | 2026년 이후의 보안 정책은? | Metadata Filtering |
| Multi-hop query | 이 함수가 의존하는 모듈의 소유자는? | Graph Traversal |

Query Router는 질문의 성격을 분류해 BM25, Vector Search, Metadata Filtering, Graph Traversal의 비중을 바꾼다. 분류가 확실하지 않다면 하나만 고르는 대신 여러 검색기를 병렬로 사용하고 결과를 합치는 방법도 있다.

라우팅 자체도 평가 대상이다. Exact query가 반복해서 Vector Search로만 전달된다면 Embedding 모델보다 Router를 먼저 고쳐야 한다.

## 5. 지식 그래프의 기본 단위

지식 그래프는 Entity를 Node로, Entity 사이의 Relation을 Edge로 표현한다. 가장 단순한 표현은 주어, 관계, 목적어로 구성된 Triple이다.

```text
LangChain       — uses       → Vector Database
Vector Database — stores     → Embedding
Embedding       — created_by → Embedding Model
```

문서에서 그래프를 만들려면 보통 세 단계가 필요하다.

1. **Entity extraction**: 문장에서 Entity 후보를 찾는다.
2. **Entity linking**: 후보가 실제로 어떤 Entity를 뜻하는지 연결한다.
3. **Relation extraction**: Entity 사이의 관계를 추출한다.

### Entity Linking이 중요한 이유

`Apple`이라는 문자열은 회사와 과일을 모두 뜻할 수 있다. 두 대상을 하나의 Entity로 합치면 다음과 같은 관계가 같은 Node에 연결된다.

```text
Apple — contains → Vitamin C
Apple — released → MacBook
```

그래프의 의미적 일관성이 깨지지 않으려면 표시 이름과 별도로 고유 Entity ID를 두고, 문맥과 타입을 이용해 대상을 연결해야 한다. 반대로 같은 회사를 `Apple`, `Apple Inc.`, `애플`이라는 세 Node로 나누는 중복 문제도 해결해야 한다.

## 6. Schema, Taxonomy, Ontology

그래프에 무엇이든 자유롭게 저장하면 관계 이름이 흔들리고 잘못된 타입 연결이 늘어난다. 그래서 허용할 Entity Type과 Relation Type을 먼저 정의할 수 있다.

```text
Company  — released     → Product
Library  — depends_on   → Library
Function — declared_in  → SourceFile
```

- **Schema**는 어떤 타입과 필드, 관계 형식을 저장할지 정의한다.
- **Taxonomy**는 개념을 상위·하위 계층으로 분류한다.
- **Ontology**는 개념의 의미, 관계, 제약과 때로는 추론 규칙까지 표현한다.

이 구분은 사용하는 기술과 조직에 따라 경계가 달라질 수 있다. 핵심은 용어 자체보다 시스템이 어떤 관계를 허용하고, 어떤 제약을 검증하며, 변경 시 호환성을 어떻게 관리하는지 명시하는 것이다.

개인 지식베이스라면 Taxonomy로 학습 주제를 탐색하고, 지식 그래프로 개념 사이의 연결을 표현하는 것부터 시작할 수 있다. 처음부터 거대한 Ontology를 만들면 유지 비용이 더 커질 수 있다.

## 7. Graph Traversal과 Multi-hop 질문

Graph Traversal은 Edge를 따라 여러 단계의 관계를 탐색하는 작업이다.

> “내 친구가 다니는 회사의 대표는 누구인가?”

```text
나 — friend_of → 친구 — works_at → 회사 — led_by → 대표
```

문서 유사도만으로는 이 관계 경로를 안정적으로 구성하기 어렵다. 그래프에서는 시작 Entity와 목표 Relation을 파악한 뒤 연결된 경로를 탐색할 수 있다.

하지만 단계가 늘어날수록 후보 경로가 급격히 많아지는 탐색 공간 폭발이 발생한다. 이를 줄이기 위해 다음 제한을 둔다.

- 이미 방문한 Node는 다시 방문하지 않는다.
- 최대 탐색 깊이를 제한한다.
- 질문에 허용된 Relation만 따라간다.
- 관련성이 낮거나 신뢰도가 낮은 경로를 중단한다.
- 반환할 경로와 이웃 수를 제한한다.

정답을 놓치지 않으려 탐색 폭을 무작정 넓히면 지연 시간과 노이즈가 커진다. 평가 데이터에서 깊이와 후보 수를 조정해야 한다.

## 8. Graph RAG는 무엇을 추가하는가

Graph RAG는 하나의 고정된 제품이나 알고리즘이 아니라, RAG 과정에서 Entity와 Relation으로 구성된 그래프를 활용하는 접근들의 묶음에 가깝다.

일반적인 구성은 다음과 같다.

```text
질문
  ↓
Vector·Keyword Search로 관련 문서와 시작 Entity 탐색
  ↓
Graph Traversal로 경로·이웃·하위 구조 탐색
  ↓
원문 근거와 그래프 구조를 Context로 구성
  ↓
LLM 답변과 출처 연결
```

Vector Search로 출발점을 찾은 뒤 그래프를 탐색하면 모든 Node를 훑지 않고도 관련 경로에 집중할 수 있다. 반대로 명확한 Entity ID가 질문에 포함돼 있다면 그래프에서 바로 시작할 수도 있다.

그래프는 원문을 대체하지 않는다. 추출 과정에서 관계가 잘못 만들어질 수 있으므로 답변에는 Triple만 넣기보다 그 관계를 뒷받침하는 원문 구간과 출처를 함께 제공해야 한다.

## 9. 관계의 신뢰도와 독립 근거

여러 문서에서 같은 관계가 반복되면 신뢰도가 높아 보인다. 하지만 하나의 원문이 여러 사이트와 보고서에 복제된 것이라면 독립적인 근거가 여러 개인 것은 아니다.

관계의 Confidence를 계산할 때는 단순 등장 횟수보다 다음을 함께 봐야 한다.

- 원출처의 품질과 최신성
- 서로 독립적인 출처의 수
- 추출 모델의 확신과 검증 결과
- 상충하는 관계의 존재
- Schema와 도메인 규칙의 일치 여부

각 Triple에는 관계만 저장하지 말고 근거를 함께 연결한다.

```text
subject, predicate, object
source_document_id
source_span
extracted_at
extractor_version
confidence
valid_from / valid_to
```

그래야 왜 이 관계가 존재하는지 설명하고, 원문이 바뀌었을 때 어떤 지식을 다시 계산할지 알 수 있다.

## 10. Provenance Graph와 Lineage Tracking

Provenance는 지식이 어디서 왔고 어떤 처리 과정을 거쳤는지 나타낸다. Provenance Graph는 원본 문서부터 Chunk, Entity, Relation, 요약, 답변까지의 파생 관계를 연결한다.

```text
원본 문서
  ├─ parsed_into → Chunk A
  │                  └─ supports → Relation 1
  └─ parsed_into → Chunk B
                     └─ summarized_into → Summary X
```

이 구조가 있으면 잘못된 원출처가 발견됐을 때 영향 범위를 추적할 수 있다.

1. 문제가 있는 원본 문서를 찾는다.
2. 그 문서에서 파생된 Chunk와 관계를 찾는다.
3. 다른 독립 근거가 없는 관계를 제거하거나 재평가한다.
4. 영향을 받은 요약과 캐시를 무효화한다.
5. 관련 평가 세트를 다시 실행한다.

이처럼 데이터가 어디서 파생됐는지 추적하는 것을 Lineage Tracking이라고 한다. 지식베이스의 삭제가 어려운 이유는 저장소가 여러 개라서가 아니라 파생 관계가 명시되지 않은 경우가 많기 때문이다.

## 11. 계층별로 평가하기

최종 답변이 틀렸다는 사실만으로는 어디를 고칠지 알기 어렵다. 지식베이스를 계층별로 평가해야 한다.

| 계층 | 확인할 내용 |
| --- | --- |
| Ingestion | 원본이 누락 없이 파싱됐는가? |
| Freshness | 변경과 삭제가 제한 시간 안에 반영됐는가? |
| Retrieval | 정답 근거가 상위 후보에 포함됐는가? |
| Entity Linking | 같은 대상을 합치고 다른 대상을 구분했는가? |
| Relation Extraction | Triple이 원문 근거와 일치하는가? |
| Graph Traversal | 필요한 경로를 찾고 불필요한 경로를 제한했는가? |
| Generation | 답변이 근거를 충실히 사용하고 출처를 연결했는가? |

운영 평가에는 삭제 검증도 포함해야 한다. 문서를 삭제한 뒤 Keyword Index, Vector Index, Graph, Cache 어디에서도 해당 지식이 검색되지 않는지 확인하는 테스트가 필요하다.

## 12. 실전 지식베이스의 전체 구조

원문의 내용을 하나의 운영 파이프라인으로 연결하면 다음과 같다.

```text
1. 원본 수집과 변경 감지
2. 구조 보존 파싱과 Chunking
3. Metadata·Embedding·Entity·Relation 생성
4. Keyword·Vector·Graph Index 저장
5. 질의 분류와 Hybrid Retrieval
6. Reranking과 Graph Traversal
7. 원문 근거를 포함한 Context 구성
8. 답변 생성과 출처 연결
9. 계층별 품질·최신성 평가
10. Provenance 기반 갱신·삭제·재평가
```

처음부터 모든 계층을 구현할 필요는 없다. 문서 RAG로 기준선을 만든 뒤, 정확한 이름 검색이 약하면 Hybrid Search를 추가하고, Multi-hop 질문이 반복해서 실패할 때 지식 그래프를 검토하는 식으로 복잡도를 늘리는 편이 좋다.

결론적으로 지식베이스의 핵심은 데이터를 많이 저장하는 것이 아니다. **어떤 근거에서 만들어진 지식인지 설명하고, 원본이 바뀌면 정확히 갱신하거나 삭제하며, 질문에 맞는 검색 구조를 선택할 수 있어야 한다.** Graph RAG의 가치는 그래프 자체가 아니라 이 연결과 추적 가능성을 답변에 활용하는 데 있다.

## 더 읽어보기

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [A Survey on Knowledge Graphs: Representation, Acquisition, and Applications](https://arxiv.org/abs/2002.00388)
- [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130)
