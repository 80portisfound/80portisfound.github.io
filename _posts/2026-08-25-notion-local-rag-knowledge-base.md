---
layout: post
title: "Notion을 개인 지식베이스로 만들기: FastAPI부터 Hybrid Search, Reranker, Local LLM까지 직접 만든 RAG 시스템"
description: "Notion 수집과 계층 보존부터 Hybrid Search, RRF, CrossEncoder, Ollama, Page-level Source Tracking, 프론트엔드까지 직접 구현한 개인용 RAG 시스템의 설계와 시행착오를 정리합니다."
date: 2026-08-25 16:30:00 +0900
tags: [RAG, Notion, FastAPI, ChromaDB, Ollama, BM25]
category: development
category_label: 개발
reading_time: 18
---

개인적으로 공부하면서 쌓이는 자료가 많아질수록 한 가지 문제가 생겼다. 자료는 분명 어디엔가 저장되어 있는데, 정작 필요할 때 원하는 내용을 다시 찾기가 어렵다는 것이다.

특히 Notion에 네트워크, LLM, 데이터 엔지니어링 등 여러 분야의 공부 내용을 계속 정리하다 보니 단순 키워드 검색만으로는 한계가 있었다.

그래서 목표를 조금 더 크게 잡았다.

> **“내 Notion 자료 전체를 지식베이스로 만들고, 자연어로 질문하면 관련 자료를 검색해 답변하고, 그 답변이 나온 실제 Notion 페이지까지 보여주는 시스템을 직접 만들어보자.”**

처음에는 FastAPI를 학습하기 위한 프로젝트로 시작했지만, 구현을 계속 확장하면서 최종적으로는 Notion 수집부터 Hybrid Search, Reranking, Local LLM 기반 답변 생성, Page-level Source Tracking, 프론트엔드까지 연결된 하나의 RAG 시스템이 되었다.

---

## 전체 시스템 구조

현재 완성한 시스템의 흐름은 다음과 같다.

```text
Notion
   ↓
Page Parsing
   ↓
Chunking
   ↓
SQLite
   ↓
Embedding
   ↓
ChromaDB
   ↓
┌──────────────────────────┐
│ Vector Search            │
│ BM25 Keyword Search      │
└────────────┬─────────────┘
             ↓
             RRF
             ↓
      CrossEncoder Reranker
             ↓
      Relevance Filtering
             ↓
         Context 생성
             ↓
        Ollama / Qwen3:8b
             ↓
      Answer + Sources
             ↓
          FastAPI
             ↓
         Frontend
```

단순히 Vector DB에 문서를 넣고 LLM에 전달하는 구조가 아니라, **검색 품질과 출처 추적을 별도의 단계로 분리했다는 점**이 이 프로젝트에서 가장 중요했다.

---

## 1. FastAPI부터 시작한 백엔드

프로젝트 초반에는 FastAPI 자체를 익히는 것이 목적이었다.

GET, POST, PUT 같은 기본 API부터 시작해 Pydantic Request/Response 모델, SQLAlchemy ORM, CRUD Layer, Router 분리, Dependency Injection, JWT 인증 등을 하나씩 구현했다.

프로젝트 구조도 점점 다음과 같이 분리했다.

```text
pybo/
├── main.py
├── database.py
├── models.py
├── auth.py
├── security.py
│
├── schemas/
├── routers/
├── crud/
│
└── services/
```

특히 이후 RAG 기능이 커지면서 `router → service → crud` 구조로 미리 나눠둔 것이 꽤 중요했다.

예를 들어 검색 API가 직접 DB와 Embedding 모델을 모두 다루는 대신,

```text
Router
→ Retrieval Service
→ Vector Store
→ Database
```

처럼 역할을 분리할 수 있었다.

처음에는 단순한 FastAPI 실습이었지만, 이 구조가 이후 전체 RAG 시스템의 기반이 됐다.

---

## 2. Notion을 데이터 소스로 연결하기

개인 지식베이스의 원본 데이터는 Notion으로 정했다.

Notion API를 통해 Database/Data Source 안의 페이지를 가져온 뒤 SQLite의 `Document`로 저장했다.

```python
class Document(Base):
    __tablename__ = "document"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str]
    source: Mapped[str]
    notion_page_id: Mapped[str]
    last_edited_time: Mapped[str]
```

`notion_page_id`를 기준으로 기존 Document를 조회하고, 이미 존재하면 갱신하고 없으면 새로 만드는 `upsert_document()` 구조도 만들었다.

또 `last_edited_time`을 비교해서 실제 Notion 문서가 수정된 경우에만 다시 인덱싱할 수 있도록 했다.

```text
Notion last_edited_time
        ↓
SQLite last_edited_time과 비교
        ↓
동일 → skip
변경 → reindex
```

강제로 전체 인덱스를 다시 만들 필요가 있을 때를 위해 `/notion/reindex`도 별도로 구현했다.

---

## 3. 생각보다 어려웠던 Notion 하위 페이지 처리

이 과정에서 가장 먼저 큰 문제가 발생했다.

예를 들어 Notion에는 이런 구조의 자료가 있었다.

```text
컴퓨터 네트워크
├── TCP
│   ├── Flow Control
│   └── Congestion Control
└── UDP
```

그런데 처음 구현한 방식에서는 상위 페이지만 읽고 있었기 때문에, 실제로 `TCP congestion control`을 검색해도 해당 내용이 검색되지 않았다.

BM25 디버그 검색으로 `TCP`를 찾아봤을 때 아예 결과가 나오지 않는 경우도 있었다.

원인은 **Notion의 `child_page` 내부에 다시 `child_page`가 존재하는 계층 구조**였다.

이를 해결하기 위해 Block의 `has_children`을 확인하면서 재귀적으로 탐색하는 구조를 만들었다.

```python
def get_all_blocks(page_id: str) -> list[dict]:
    blocks = get_block_children(page_id)

    all_blocks = []

    for block in blocks:
        all_blocks.append(block)

        if block.get("has_children"):
            child_blocks = get_all_blocks(block["id"])
            all_blocks.extend(child_blocks)

    return all_blocks
```

이 문제를 해결한 뒤 `TCP` 키워드가 포함된 Chunk가 정상적으로 검색되기 시작했다.

이 경험을 통해 RAG에서 검색 모델만큼 중요한 것이 **원본 데이터를 얼마나 정확하게 수집하고 구조화하는가**라는 점을 알게 됐다.

---

## 4. 문서를 Chunk로 분할하기

Notion Page 전체를 그대로 Embedding하면 검색 단위가 너무 커지기 때문에 문서를 여러 Chunk로 분리했다.

초기 Chunking은 단순한 Character 기반 Sliding Window 방식으로 구성했다.

```python
def split_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:

    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(text), step):
        chunks.append(text[i:i + chunk_size])

    return chunks
```

기본 설정은:

```text
chunk_size = 500
overlap = 100
```

이다.

Overlap을 둔 이유는 Chunk 경계에서 문맥이 끊기는 문제를 어느 정도 줄이기 위해서였다.

각 Chunk는 SQLite에 다음과 같이 저장했다.

```text
Document
  ↓
Chunk 0
Chunk 1
Chunk 2
...
```

`chunk_index`를 저장해서 검색 결과가 너무 가까운 인접 Chunk일 경우 중복을 제거하는 데도 사용했다.

---

## 5. Multilingual E5 + ChromaDB로 Vector Search 구축

Embedding 모델은 다음 모델을 사용했다.

```text
intfloat/multilingual-e5-small
```

E5 계열 모델의 입력 형식에 맞춰 문서와 Query Prefix를 구분했다.

```text
passage: 문서 내용
query: 사용자 질문
```

구현은 다음과 같은 형태다.

```python
def embed_text(text: str) -> list[float]:
    text = f"passage: {text}"
    embedding = model.encode(text)

    return embedding.tolist()


def embed_query(query: str) -> list[float]:
    query = f"query: {query}"
    embedding = model.encode(query)

    return embedding.tolist()
```

생성된 Embedding은 ChromaDB에 저장했다.

```text
SQLite
→ Chunk
→ Embedding
→ ChromaDB
```

Chroma에는 Vector뿐 아니라 Chunk를 추적하기 위한 Metadata도 같이 저장했다.

```python
{
    "chunk_id": chunk.id,
    "document_id": chunk.document_id,
    "chunk_index": chunk.chunk_index,
}
```

이제 자연어 질문을 Embedding한 뒤 Chroma에서 의미적으로 가까운 Chunk를 찾을 수 있게 됐다.

---

## 6. Vector Search만으로는 부족했다

Vector Search를 구현한 뒤 한 가지 한계가 보였다.

자연어 의미 검색에는 강했지만,

```text
TCP
CWND
CUBIC
```

같은 정확한 전문 용어를 찾을 때는 반드시 Vector Search가 가장 좋은 결과를 반환하지 않았다.

그래서 **BM25를 추가했다.**

`rank-bm25`의 `BM25Okapi`를 사용했고, SQLite의 전체 Chunk를 대상으로 점수를 계산했다.

```python
tokenized_chunks = [
    chunk.content.lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)

tokenized_query = query.lower().split()

scores = bm25.get_scores(tokenized_query)
```

이제 검색 시스템은 두 가지 성격을 모두 가지게 됐다.

```text
Vector Search
→ 의미적으로 비슷한 내용 검색

BM25
→ 정확한 단어와 용어 검색
```

---

## 7. Vector와 BM25를 어떻게 합칠 것인가

문제는 두 검색의 Score 체계가 완전히 다르다는 것이다.

Vector Search에서는 Distance가 낮을수록 좋고,

```text
distance ↓ = 관련성 ↑
```

BM25는 Score가 높을수록 좋다.

```text
score ↑ = 관련성 ↑
```

따라서 단순히 두 점수를 더하는 방식은 사용할 수 없었다.

그래서 **Reciprocal Rank Fusion, RRF**를 적용했다.

RRF는 Score 자체가 아니라 각 검색 시스템에서의 **순위**를 사용한다.

```text
Vector rank = 1
BM25 rank = 3

→ 두 순위를 기반으로 최종 점수 계산
```

기본 `k=60`으로 다음 구조를 사용했다.

```text
1 / (60 + vector_rank)
+
1 / (60 + keyword_rank)
```

한 검색에서만 발견된 Chunk도 그대로 사용할 수 있다.

```text
Vector에만 있음
→ Vector rank만 사용

BM25에만 있음
→ BM25 rank만 사용

둘 다 있음
→ 두 점수 합산
```

이를 통해 Vector와 BM25의 결과를 하나의 후보군으로 통합했다.

---

## 8. Retrieval과 Reranking을 분리했다

RRF까지 구현한 뒤에도 한 가지 문제가 남았다.

검색은 **많은 후보를 빠르게 찾는 것**에는 좋지만, 정말 질문에 가장 적합한 순서를 완벽하게 정하는 것은 어려웠다.

그래서 Retrieval 이후에 별도의 Reranker를 추가했다.

사용한 모델은:

```text
BAAI/bge-reranker-v2-m3
```

CrossEncoder는 Query와 Chunk를 동시에 입력받는다.

```python
pairs = [
    (query, result["content"])
    for result in results
]

scores = model.predict(pairs)
```

그리고 검색 결과에:

```python
"rerank_score"
```

를 추가한 뒤 다시 정렬했다.

전체 검색 과정은 결국 다음과 같이 됐다.

```text
Query
↓
Vector Search ───┐
                 ├─ Candidate
BM25 ────────────┘
↓
RRF
↓
CrossEncoder
↓
Final Top-K
```

즉 **Retrieval은 Recall을 확보하고, Reranker가 Precision을 높이는 구조**로 만들었다.

---

## 9. “검색 결과가 있다”와 “정답이 있다”는 다르다

Hybrid Search는 어떤 질문을 입력하더라도 가장 가까운 Top-K를 찾아낸다.

예를 들어 내 지식베이스에 역사 자료가 전혀 없는데도,

> 세종대왕의 키는 몇 cm야?

라고 질문하면 가장 덜 관련 없는 Java나 AI Chunk라도 결과로 반환될 수 있다.

그래서 Reranker Score를 이용한 Threshold를 추가했다.

실제로 테스트한 결과는 꽤 명확했다.

| 질문 | Top rerank score |
| --- | ---: |
| TCP 혼잡 제어가 뭐야? | 약 0.998 |
| TCP와 UDP의 차이가 뭐야? | 약 0.914 |
| 혼잡 제어가 왜 필요한데? | 약 0.890 |
| 세종대왕의 키는 몇 cm야? | 약 0.0056 |

1차 실험에서는 Threshold를 다음과 같이 설정했다.

```python
min_rerank_score = 0.1
```

현재 `/ask` API의 Query Parameter 기본값은 `0.0`이므로 이 차단 기준을 기본 동작으로 적용하려면 요청에 `0.1`을 명시하거나 API 기본값을 변경해야 한다.

Threshold보다 낮은 Chunk는 LLM에 전달하지 않는다.

```python
relevant_results = [
    result
    for result in results
    if result["rerank_score"] >= min_rerank_score
]
```

모든 결과가 제거되면:

```json
{
  "answer": "제공된 자료에서 답을 찾을 수 없습니다.",
  "sources": []
}
```

를 즉시 반환한다.

이 방식의 중요한 점은 **LLM에게 판단을 전적으로 맡기기 전에 Retrieval Layer에서 먼저 차단한다는 것**이다.

---

## 10. 이제 검색 결과를 Qwen에게 전달한다

검색 시스템이 어느 정도 안정화된 뒤 실제 RAG Generation을 붙였다.

OpenAI API 대신 로컬 환경에서 Ollama를 사용했고 모델은:

```text
qwen3:8b
```

을 사용했다.

검색 결과는 먼저 Context 형태로 만든다.

```text
[자료 1]
TCP 혼잡 제어는 ...

[자료 2]
Congestion Window는 ...
```

그리고 다음과 같은 규칙을 포함한 Prompt를 생성했다.

```text
다음 자료만을 근거로 질문에 답하세요.
자료에 없는 내용은 추측하거나 만들어내지 마세요.
자료만으로 답할 수 없다면
'제공된 자료에서 답을 찾을 수 없습니다.'
라고 답하세요.
```

최종적으로:

```python
response = chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
)
```

을 통해 답변을 생성한다.

이제 하나의 질문은 다음 전체 파이프라인을 거치게 됐다.

```text
Question
↓
Hybrid Retrieval
↓
RRF
↓
Reranker
↓
Threshold
↓
Context
↓
Prompt
↓
Qwen3:8b
↓
Answer
```

---

## 11. RAG에서 답변만큼 중요했던 Source

처음 RAG를 연결했을 때는 답변 자체는 잘 나왔다.

하지만 Source가 다음처럼 나왔다.

```text
Document 17
컴퓨터 네트워크
```

문제가 있었다.

내 Notion에는 `컴퓨터 네트워크` 아래 수많은 하위 페이지가 있는데, 실제 답변이 어느 페이지에서 나온 것인지 알 수 없었다.

그래서 **Page-level Source Tracking**을 구현하기로 했다.

이 부분이 프로젝트 후반부에서 가장 큰 구조 변경 중 하나였다.

---

## 12. Notion의 Page Hierarchy 자체를 보존하기

기존에는 모든 하위 Page의 내용을 하나의 문자열로 합쳤다.

이를 새로운 `get_page_sections()` 구조로 바꿨다.

각 Page를 다음 형태로 유지한다.

```python
{
    "page_id": "...",
    "page_title": "TCP congestion Control",
    "page_path": "컴퓨터 네트워크 > TCP congestion Control",
    "content": "..."
}
```

재귀적으로 `child_page`를 따라가면서 Parent Path를 누적한다.

```python
current_path = parent_path + [page_title]
```

이를 통해:

```text
컴퓨터 네트워크
```

뿐 아니라:

```text
컴퓨터 네트워크
> TCP
> Congestion Control
```

같은 실제 Notion 계층을 유지할 수 있게 됐다.

---

## 13. Chunk 자체가 자신의 출처를 기억하도록 변경

Page hierarchy를 읽는 것만으로는 충분하지 않았다.

검색 단위는 결국 Chunk이기 때문에 Chunk 자체가 어느 Page에서 나왔는지 알아야 한다.

그래서 `Chunk` 모델을 확장했다.

```text
Chunk
├── id
├── document_id
├── chunk_index
├── content
├── page_id
├── page_title
└── page_path
```

추가된 필드는 다음과 같다.

```python
page_id
page_title
page_path
```

기존 SQLite에는 이미 많은 Chunk가 저장되어 있었기 때문에 `nullable=True`로 컬럼을 추가하고 별도의 Migration도 수행했다.

또 새롭게 생성되는 Chunk는 항상 Page Metadata를 가지도록 `ChunkCreate`도 변경했다.

---

## 14. Page별로 Chunking하는 구조로 변경

기존에는:

```text
전체 Notion Document
↓
하나의 긴 Text
↓
Chunking
```

이었다면 이후에는:

```text
Notion Document
↓
Page A
Page B
Page C
↓
각 Page별 Chunking
↓
Chunk + Page Metadata
```

로 변경했다.

`sync_notion_sections()`가 각 Page를 순회하면서 Chunk를 만들고,

```python
page_id
page_title
page_path
```

를 함께 저장한다.

또 Page가 바뀔 때마다 `chunk_index`가 0으로 초기화되지 않도록 `start_index`를 사용했다.

```text
Page A → 0, 1, 2
Page B → 3, 4
Page C → 5, 6, 7
```

이 구조 덕분에 기존 인접 Chunk 판단 로직도 그대로 유지할 수 있었다.

---

## 15. SQLite뿐 아니라 Chroma까지 Metadata를 전달

Page Metadata를 DB에만 저장하면 Vector Search에서는 사용할 수 없다.

따라서 Chroma의 Metadata에도 동일한 정보를 저장했다.

```python
{
    "chunk_id": chunk.id,
    "document_id": chunk.document_id,
    "chunk_index": chunk.chunk_index,
    "page_id": chunk.page_id,
    "page_title": chunk.page_title,
    "page_path": chunk.page_path,
}
```

그리고 `search_vectors()`가 이 Metadata를 다시 반환하게 했다.

BM25 결과에도 동일한 필드를 넣었다.

결국:

```text
Vector Search ─┐
               ├→ Page Metadata 유지
BM25 ──────────┘
       ↓
      RRF
       ↓
   Reranker
       ↓
     RAG
```

전체 Pipeline 동안 Source 정보가 사라지지 않는다.

기존 데이터를 새로운 구조로 바꾸기 위해 전체 Reindex도 수행했고, 당시 19개의 기존 Document를 모두 새 Page-level 구조로 다시 인덱싱했다.

---

## 16. 최종적으로 Source가 이렇게 바뀌었다

초기:

```json
{
  "document_id": 17,
  "title": "컴퓨터 네트워크"
}
```

현재:

```json
{
  "document_id": 17,
  "title": "컴퓨터 네트워크",
  "page_id": "...",
  "page_title": "TCP congestion Control",
  "page_path": "컴퓨터 네트워크 > TCP congestion Control",
  "page_url": "...",
  "content": "..."
}
```

`page_id`를 이용해 실제 Notion 하위 페이지 URL도 생성했다.

따라서 사용자는 답변을 읽은 뒤:

> **이 정보가 실제 내 Notion의 어디에 있는가?**

를 바로 확인할 수 있다.

개인 지식베이스에서는 이 기능이 단순 Citation 이상으로 중요하다고 생각한다. RAG가 답을 대신 만들어주는 데서 끝나는 것이 아니라, **원래 공부했던 자료로 다시 돌아갈 수 있기 때문**이다.

---

## 17. FastAPI의 `/ask` 하나로 전체 Pipeline 실행

최종적으로 사용자용 API는 매우 단순해졌다.

```text
GET /ask
```

입력:

```text
query
top_k
min_rerank_score
```

API 내부에서는:

```text
answer_question()
↓
Hybrid Search
↓
RRF
↓
Reranker
↓
Threshold Filtering
↓
Qwen
↓
Answer + Sources
```

가 전부 자동 실행된다.

관련 자료가 있으면 답변과 실제 Notion Page가 반환되고, 관련 자료가 없으면:

```json
{
  "answer": "제공된 자료에서 답을 찾을 수 없습니다.",
  "sources": []
}
```

를 반환한다.

---

## 18. 마지막으로 프론트엔드까지 연결했다

백엔드와 RAG Pipeline을 완성한 뒤에는 사용자 입장에서 직접 사용할 수 있도록 프론트엔드도 구현했다.

이제 Swagger에서 직접 `/ask`를 호출하는 개발용 형태를 넘어, 프론트엔드에서 질문을 입력하면 백엔드 RAG Pipeline이 실행되고 답변과 Source를 확인할 수 있는 하나의 애플리케이션 형태가 됐다.

특히 Source에 `page_title`, `page_path`, `page_url`이 포함되어 있기 때문에 검색 결과에서 실제 Notion 원문으로 다시 이동하는 UX도 만들 수 있게 됐다.

결국 처음 목표였던:

> **“내가 쌓아둔 자료에 자연어로 질문하고, 답을 얻은 뒤 실제 원문까지 돌아가는 개인용 지식베이스”**

를 하나의 End-to-End 시스템으로 구현했다.

---

## 구현하면서 가장 크게 배운 점

이 프로젝트를 시작하기 전에는 RAG를 비교적 단순하게 생각했다.

```text
Document
→ Embedding
→ Vector DB
→ LLM
```

정도로 생각하기 쉬웠다.

하지만 실제로 만들어보니 가장 많은 고민이 필요했던 부분은 LLM 호출 자체가 아니었다.

**문서를 어떻게 파싱할지, 검색 단위를 어떻게 정할지, Keyword Search와 Semantic Search를 어떻게 결합할지, 관련 없는 검색 결과를 어떻게 차단할지, 출처를 어느 수준까지 추적할지** 같은 Retrieval과 Data Pipeline 설계가 훨씬 중요했다.

특히 이번 프로젝트에서는 세 가지를 크게 배웠다.

첫째, **RAG의 품질은 입력 데이터 구조에서부터 결정된다.** Notion 하위 페이지를 제대로 읽지 못했을 때 아무리 좋은 Embedding 모델을 사용해도 TCP 자료를 검색할 수 없었다.

둘째, **한 종류의 검색만으로 충분하지 않았다.** Vector Search는 의미 검색에 강했고 BM25는 정확한 기술 용어 검색에 강했다. 두 검색을 RRF로 결합하고 다시 CrossEncoder로 Reranking하면서 훨씬 안정적인 Retrieval Pipeline을 만들 수 있었다.

셋째, **RAG에는 “모른다”는 판단이 필요하다.** 검색 시스템은 항상 무언가를 반환한다. 하지만 가장 비슷한 문서가 존재한다는 것과 그 문서가 질문의 답이라는 것은 다르다. Reranker Threshold를 도입하면서 관련성이 낮은 검색 결과가 LLM으로 넘어가는 것을 차단할 수 있었다.

---

## 현재 프로젝트의 최종 구조

지금까지 구현한 시스템을 한 번에 표현하면 다음과 같다.

```text
                    ┌──────────────┐
                    │    Notion    │
                    └──────┬───────┘
                           ↓
                  Page Hierarchy Parser
                           ↓
                     Page Chunking
                           ↓
              ┌────────────┴────────────┐
              │                         │
           SQLite                    Embedding
        Metadata DB             multilingual-e5-small
                                        ↓
                                     ChromaDB
                                        │
                    ┌───────────────────┴──────────────────┐
                    ↓                                      ↓
              Vector Search                            BM25 Search
                    │                                      │
                    └───────────────────┬──────────────────┘
                                        ↓
                                       RRF
                                        ↓
                         BGE CrossEncoder Reranker
                                        ↓
                         min_rerank_score Filtering
                                        ↓
                                  Context Builder
                                        ↓
                                   Prompt Builder
                                        ↓
                              Ollama / Qwen3:8b
                                        ↓
                              Answer + Page Source
                                        ↓
                                     FastAPI
                                        ↓
                                    Frontend
```

처음에는 FastAPI를 공부하기 위해 만들기 시작한 프로젝트였지만, 결과적으로 **데이터 수집 → 저장 → 검색 → Ranking → LLM Generation → Source Tracking → API → Frontend**까지 직접 연결해보는 프로젝트가 됐다.

---

## 앞으로 개선할 것

현재 시스템은 실제로 사용할 수 있는 수준까지 연결했지만, 다음 단계부터는 **“동작하는가?”보다 “얼마나 잘 동작하는가?”**를 다루려고 한다.

앞으로 추가하고 싶은 부분은 다음과 같다.

* Retrieval Evaluation: Hit@K, MRR 등으로 검색 품질 측정
* Rerank Threshold를 소수의 테스트가 아니라 Evaluation Dataset을 이용해 결정
* 질문, 검색 결과, Rerank Score, 최종 답변 Logging
* Retrieval Failure와 Generation Failure 분리
* 한국어 BM25 Tokenization 개선
* Chunking 전략 비교
* Notion 증분 업데이트 고도화
* Retrieval 및 Generation 품질 Dashboard 구축
* 실제 사용 로그를 이용한 Failure Case 분석

지금까지는 **RAG 시스템을 만드는 과정**이었다면, 다음부터는 이 RAG를 하나의 실제 소프트웨어 시스템처럼 **평가하고 관찰하고 개선하는 단계**로 넘어가려고 한다.

---

## 마무리

이번 프로젝트에서 가장 의미 있었던 점은 특정 라이브러리를 사용해봤다는 것보다 **RAG의 각 Layer를 직접 하나씩 만들어봤다는 것**이다.

```text
Notion API
→ Parsing
→ Chunking
→ SQL
→ Embedding
→ Vector DB
→ BM25
→ Hybrid Search
→ RRF
→ Reranking
→ Relevance Filtering
→ Local LLM
→ Citation
→ FastAPI
→ Frontend
```

처음에는 각각 별개의 기술처럼 보였지만, 하나의 질문이 들어와 최종 답변과 근거가 반환되기까지 직접 연결하면서 각 기술이 왜 필요한지 훨씬 명확하게 이해할 수 있었다.

그리고 현재 시스템은 단순한 챗봇이라기보다,

> **내가 학습한 자료를 다시 찾고, 연결하고, 원문으로 돌아갈 수 있게 만드는 개인용 Knowledge Retrieval System**

에 더 가깝다.

앞으로는 여기에 Evaluation, Logging, Observability를 추가해서 단순히 “잘 되는 것처럼 보이는 RAG”가 아니라 **실제로 성능을 측정하고 개선할 수 있는 RAG 시스템**으로 발전시켜볼 생각이다.
