---
layout: post
title: "Notion을 로컬 RAG 지식베이스로 만들기: 검색부터 출처 추적까지"
description: "Notion 문서를 동기화하고 청크·임베딩·하이브리드 검색·CrossEncoder 재정렬·Ollama 답변 생성·페이지 단위 출처 추적을 연결한 로컬 RAG 프로젝트의 설계와 시행착오를 정리합니다."
date: 2026-08-25 16:30:00 +0900
tags: [RAG, Notion, FastAPI, ChromaDB, Ollama]
category: development
category_label: 개발
reading_time: 15
---

Notion에 자료가 쌓일수록 새로운 문제가 생긴다. 분명 기록해 둔 내용인데 어느 페이지에 있는지 기억나지 않고, 검색어를 정확히 모르면 찾기 어렵다. 그래서 Notion 문서를 로컬로 동기화한 뒤 자연어로 질문하고, 답변의 근거 페이지까지 확인할 수 있는 지식베이스를 만들었다.

완성된 파이프라인은 다음과 같다.

```text
Notion
  → 페이지·하위 페이지 동기화
  → 청킹과 메타데이터 저장
  → 다국어 임베딩과 ChromaDB 색인
  → 벡터 검색 + BM25
  → RRF 결합 + CrossEncoder 재정렬
  → Ollama의 Qwen3:8b로 답변 생성
  → 답변 + 원문 페이지 출처 반환
```

이 글은 기능 목록보다 개인 프로젝트 `pybo`가 어떤 문제를 만나며 이 구조로 발전했는지에 초점을 맞춘다. 저장소는 현재 비공개이므로 이 글에서는 핵심 설계와 코드만 설명한다.

## 1. 문서와 청크를 분리해서 저장하기

RAG에서 원본 문서 전체를 그대로 검색 대상으로 삼기는 어렵다. 긴 문서 하나를 임베딩하면 서로 다른 주제가 하나의 벡터에 섞이고, LLM에 전달할 컨텍스트도 불필요하게 커진다. 그래서 데이터 모델을 `Document`와 `Chunk`로 나눴다.

```text
Document 1 ── N Chunk

Document: Notion 최상위 문서의 제목, URL, 수정 시각
Chunk: 검색에 사용할 본문 조각과 원문 페이지 정보
```

`Document`에는 Notion 페이지 ID와 마지막 수정 시각을 저장한다. 같은 페이지가 이미 존재하는지 판단하고, 내용이 바뀌었을 때만 다시 색인하기 위해서다. `Chunk`에는 본문과 순서뿐 아니라 다음 메타데이터도 함께 둔다.

```python
class Chunk(Base):
    __tablename__ = "chunk"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("document.id"),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text(), nullable=False)
    chunk_index: Mapped[int] = mapped_column(nullable=False)
    page_id: Mapped[str | None] = mapped_column(String(200))
    page_title: Mapped[str | None] = mapped_column(String(200))
    page_path: Mapped[str | None] = mapped_column(String(200))
```

처음에는 `document_id`, `content`, `chunk_index`만 저장했다. 검색은 가능했지만, 답변의 근거가 최상위 문서 아래 어느 하위 페이지에서 왔는지 알 수 없었다. `page_id`, `page_title`, `page_path`는 이 문제를 해결하기 위해 나중에 추가한 필드다.

## 2. Notion을 증분 동기화하기

Notion API에서 데이터 소스의 페이지 목록을 가져온 뒤 각 페이지의 메타데이터를 로컬 DB에 upsert한다. 핵심 기준은 `last_edited_time`이다.

```python
def needs_update(db_document, page_metadata: dict) -> bool:
    if db_document is None:
        return True
    return (
        db_document.last_edited_time
        != page_metadata.get("last_edited_time")
    )
```

일반 동기화는 새 문서를 생성하고 변경된 문서만 다시 처리한다. 전체 색인을 새 구조로 바꿨거나 임베딩 모델을 교체했을 때는 강제 재색인 엔드포인트를 사용한다.

```text
POST /notion/sync     # 생성·변경된 문서만 처리
POST /notion/reindex  # 모든 문서를 강제로 다시 색인
```

문서를 다시 색인할 때는 SQLite의 청크와 ChromaDB의 벡터를 따로 생각하면 안 된다. 한쪽만 갱신하면 존재하지 않는 청크를 벡터 검색이 반환하거나, 수정 전 내용이 계속 검색될 수 있다. 프로젝트에서는 다음 순서로 두 저장소를 함께 교체한다.

```text
기존 Chroma 벡터 삭제
  → 기존 SQLite 청크 삭제
  → 새 청크 생성
  → 새 임베딩 생성
  → ChromaDB에 저장
```

Notion API 호출에는 네트워크 지연과 `429`, `5xx` 응답도 발생할 수 있다. 연결 제한 시간을 분리하고 재시도 가능한 상태 코드에 지수 백오프를 적용해 대량의 하위 페이지를 읽을 때의 안정성을 높였다. 다만 실제 서비스에서는 Notion API의 페이지네이션도 끝까지 처리해야 한다. 현재처럼 응답의 첫 페이지만 읽는 구현은 데이터가 많아지면 일부 블록을 놓칠 수 있다.

## 3. 하위 페이지의 경계를 보존하기

초기 구현은 최상위 페이지 아래의 모든 블록을 재귀적으로 모은 뒤 하나의 긴 문자열로 합쳤다. 텍스트 자체는 수집되지만 페이지 경계가 사라진다.

```text
컴퓨터 네트워크
├── TCP
│   ├── 혼잡 제어
│   └── 흐름 제어
└── UDP
```

이 구조를 평평한 문자열로 만들면 검색 결과가 `컴퓨터 네트워크`에서 왔다는 사실만 남는다. 사용자가 실제로 보고 싶은 것은 `컴퓨터 네트워크 > TCP > 혼잡 제어`라는 구체적인 위치다.

이를 해결하기 위해 재귀 탐색의 반환 단위를 블록 목록이 아니라 페이지 섹션으로 바꿨다.

```python
{
    "page_id": "...",
    "page_title": "혼잡 제어",
    "page_path": "컴퓨터 네트워크>TCP>혼잡 제어",
    "content": "...",
}
```

각 섹션을 독립적으로 청킹하고 모든 청크에 페이지 메타데이터를 붙인다. 페이지가 바뀔 때 `chunk_index`를 0으로 되돌리지 않고 문서 전체에서 연속된 순서를 유지한다. 그 결과 인접 청크 판별과 원문 위치 추적을 동시에 할 수 있다.

## 4. 겹치는 청크와 다국어 임베딩

현재 청킹은 문자 수를 기준으로 500자씩 자르고 100자를 겹친다.

```python
def split_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    step = chunk_size - overlap
    return [
        text[i : i + chunk_size]
        for i in range(0, len(text), step)
    ]
```

오버랩은 청크 경계에서 문맥이 끊기는 문제를 줄인다. 반면 같은 내용이 여러 후보에 반복될 수 있으므로, 벡터 검색 결과를 고를 때 같은 문서의 인접한 `chunk_index`가 이미 선택되었다면 제외하는 로직도 추가했다.

임베딩 모델은 `intfloat/multilingual-e5-small`을 사용한다. E5 계열의 학습 방식에 맞게 문서에는 `passage:`, 질문에는 `query:` 접두사를 붙인다.

```python
def embed_texts(texts: list[str]) -> list[list[float]]:
    passages = [f"passage: {text}" for text in texts]
    return model.encode(passages).tolist()


def embed_query(query: str) -> list[float]:
    return model.encode(f"query: {query}").tolist()
```

생성한 벡터는 ChromaDB에 저장한다. SQLite가 문서와 청크의 기준 데이터라면 ChromaDB는 검색을 위한 파생 인덱스다. `chunk_id`뿐 아니라 페이지 메타데이터까지 Chroma에 함께 넣어야 검색 파이프라인의 어느 단계에서도 출처가 유실되지 않는다.

## 5. 벡터 검색만으로는 부족했다

벡터 검색은 표현이 달라도 의미가 비슷한 문장을 찾는 데 강하다. 반대로 약어, 함수명, 고유명사처럼 정확한 문자열이 중요한 질문에서는 키워드 검색이 더 잘 작동할 수 있다. 그래서 두 방식을 함께 사용했다.

- 벡터 검색: `multilingual-e5-small`과 ChromaDB로 의미적 유사성을 검색
- 키워드 검색: BM25로 질문과 같은 단어가 포함된 청크를 검색

서로 다른 두 점수는 척도가 다르므로 원점수를 바로 더하지 않았다. 각 검색 결과의 순위만 이용하는 RRF(Reciprocal Rank Fusion)로 결합한다.

```python
def calculate_rrf_score(
    vector_rank: int | None,
    keyword_rank: int | None,
    k: int = 60,
) -> float:
    score = 0.0
    if vector_rank is not None:
        score += 1 / (k + vector_rank)
    if keyword_rank is not None:
        score += 1 / (k + keyword_rank)
    return score
```

두 검색에서 모두 높은 순위를 받은 청크는 더 높은 RRF 점수를 얻는다. 한쪽에만 등장한 청크도 버리지 않는다. 먼저 넓게 후보를 모은다는 점이 중요하다.

## 6. CrossEncoder로 최종 순위를 다시 매기기

벡터 검색과 BM25는 후보를 빠르게 찾는 역할에 적합하지만, 질문과 각 후보의 관련성을 세밀하게 비교하는 데는 한계가 있다. RRF 상위 후보를 `BAAI/bge-reranker-v2-m3` CrossEncoder에 전달해 최종 순위를 다시 계산한다.

```text
Vector top 3K ─┐
               ├→ RRF → top 3K → CrossEncoder → top K
BM25 top 3K ───┘
```

CrossEncoder는 질문과 청크를 한 쌍으로 함께 읽기 때문에 임베딩 간 거리보다 정교한 관련도 점수를 낼 수 있다. 대신 모든 문서를 대상으로 실행하기에는 비싸다. 빠른 검색기로 후보를 줄이고 느린 재정렬기를 마지막에 사용하는 이유다.

재정렬 점수는 답변 거부에도 활용할 수 있다. 노트에서 수행한 실험에서는 관련 질문의 1위 점수가 약 `0.89~0.998`, 지식베이스와 관계없는 질문은 약 `0.0056`으로 구분됐다. 이를 바탕으로 `0.1`을 초기 임곗값으로 실험했다.

이 값은 보편적인 기준이 아니다. 모델, 문서, 질문 분포가 바뀌면 점수 분포도 바뀐다. 현재 API는 `min_rerank_score`를 요청 파라미터로 노출하므로 실제 질의 데이터로 정답률과 거부율을 측정하며 조정해야 한다.

## 7. 검색 결과를 로컬 LLM의 답변으로 바꾸기

검색이 끝나면 관련 청크를 번호가 붙은 자료로 조합한다.

```text
[자료 1]
...

[자료 2]
...
```

프롬프트에는 세 가지 규칙을 둔다.

1. 제공된 자료만 근거로 답한다.
2. 자료에 없는 내용은 추측하지 않는다.
3. 자료로 답할 수 없으면 찾을 수 없다고 명시한다.

답변 모델은 로컬 Ollama의 `qwen3:8b`를 사용했다. 외부 LLM API로 개인 노트를 보내지 않고 로컬에서 처리할 수 있다는 점이 이 프로젝트의 중요한 선택이었다.

```text
GET /ask?query=TCP 혼잡 제어가 뭐야?&top_k=5
```

관련도 임곗값을 넘은 청크가 하나도 없다면 LLM을 호출하지 않고 바로 다음 응답을 반환한다.

```json
{
  "answer": "제공된 자료에서 답을 찾을 수 없습니다.",
  "sources": []
}
```

이 방식이 환각을 완전히 제거하는 것은 아니다. 검색된 자료를 모델이 잘못 해석할 수도 있고, 임곗값이 낮으면 관계없는 컨텍스트가 들어올 수 있다. 다만 근거가 없을 때 생성을 시작하지 않는 명확한 방어선은 된다.

## 8. 답변보다 중요한 출처 추적

RAG 답변을 신뢰하려면 생성된 문장만 보여줘서는 안 된다. 어떤 원문을 근거로 했는지 사용자가 직접 확인할 수 있어야 한다.

초기에는 출처 중복 제거 기준으로 `document_id`를 사용했다. 그러면 같은 최상위 문서에 포함된 여러 하위 페이지가 모두 하나의 출처로 합쳐진다. 기준을 `page_id`로 바꾸자 서로 다른 하위 페이지를 각각 보여줄 수 있게 됐다.

```json
{
  "document_id": 17,
  "title": "컴퓨터 네트워크",
  "page_title": "TCP congestion Control",
  "page_path": "컴퓨터 네트워크>TCP congestion Control",
  "page_url": "https://www.notion.so/...",
  "content": "검색에 사용된 문단"
}
```

이 메타데이터는 SQLite에서 시작해 ChromaDB, 벡터 검색, BM25, RRF, reranker, 최종 응답까지 끊기지 않고 전달되어야 한다. 중간 단계 하나라도 빠뜨리면 답변은 생성되지만 출처 링크는 잘못될 수 있다. 출처 추적은 응답 직전에 덧붙이는 UI 기능이 아니라 색인 설계부터 관통하는 데이터 모델의 문제였다.

## 9. FastAPI로 파이프라인을 관찰 가능하게 만들기

최종 사용자에게 필요한 것은 `/ask`지만, 개발 과정에서는 각 단계를 따로 검사할 수 있는 엔드포인트가 유용했다.

| Endpoint | 역할 |
| --- | --- |
| `POST /notion/sync` | 변경된 Notion 문서 동기화 |
| `POST /notion/reindex` | 전체 문서 강제 재색인 |
| `POST /vector/backfill` | 기존 청크의 벡터 생성 |
| `GET /search` | 거리 필터가 있는 벡터 검색 |
| `GET /search/hybrid` | 하이브리드 검색과 재정렬 |
| `GET /ask` | 답변과 페이지 출처 생성 |

예를 들어 원하는 내용이 검색되지 않았을 때 곧바로 LLM을 의심하지 않고 다음 순서로 범위를 좁힐 수 있다.

```text
Notion 원문이 수집됐는가?
  → SQLite에 청크가 있는가?
  → Chroma에 벡터가 있는가?
  → 벡터 검색과 BM25 중 어디에서 놓쳤는가?
  → reranker가 관련 후보를 제거했는가?
  → 최종 프롬프트에 청크가 들어갔는가?
```

실제로 `TCP`가 검색되지 않았던 문제도 임베딩 모델보다 Notion 하위 블록 수집에 원인이 있었다. 파이프라인을 단계별로 관찰할 수 있게 만든 덕분에 원인을 빠르게 분리할 수 있었다.

## 10. 지금 구조의 한계와 다음 개선점

현재 구현은 개인용 로컬 지식베이스의 전체 흐름을 검증하기에는 충분하지만 개선할 부분도 분명하다.

- 문자 수 기반 청킹은 제목, 문단, 코드 블록 같은 의미 구조를 고려하지 않는다.
- BM25가 요청마다 전체 청크를 다시 읽고 인덱스를 만든다.
- 단순 공백 토큰화는 한국어 형태 변화에 약하다.
- Notion 목록·블록 조회의 페이지네이션을 완전히 처리해야 한다.
- 동기화 중 오류가 나면 SQLite와 ChromaDB가 잠시 불일치할 수 있으므로 작업 단위와 복구 전략이 필요하다.
- reranker 임곗값은 소수의 예시가 아니라 별도 평가 데이터셋으로 보정해야 한다.
- 모델을 모듈 import 시 즉시 로드하므로 시작 시간과 테스트 격리를 개선할 여지가 있다.

다음 단계에서는 제목과 문단을 보존하는 구조 기반 청킹, 캐시된 BM25 인덱스, 검색 평가셋을 우선 적용할 생각이다. 답변 품질을 높이기 전에 검색 단계의 재현율과 출처 정확도를 수치로 확인하는 것이 먼저다.

## 마치며

이 프로젝트에서 가장 크게 배운 점은 RAG가 단순히 "문서를 임베딩하고 LLM에 넣는 기술"이 아니라는 것이다. 실제로는 동기화, 데이터 모델, 검색, 재정렬, 거부 정책, 출처 추적이 맞물린 시스템이다.

특히 페이지 경계를 보존하고 메타데이터를 파이프라인 끝까지 전달하는 설계가 중요했다. 답을 그럴듯하게 만드는 것보다 사용자가 원문으로 돌아가 검증할 수 있게 만드는 것이 지식베이스의 신뢰도를 더 크게 높여 줬다.

## 참고 자료

- [Notion API: Working with page content](https://developers.notion.com/docs/working-with-page-content)
- [Sentence Transformers: Retrieve & Re-Rank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
- [ChromaDB documentation](https://docs.trychroma.com/)
- [Ollama Python library](https://github.com/ollama/ollama-python)
- [Reciprocal Rank Fusion 논문](https://dl.acm.org/doi/10.1145/1571941.1572114)
