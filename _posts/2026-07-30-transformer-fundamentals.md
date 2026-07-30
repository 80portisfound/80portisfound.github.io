---
layout: post
title: "Transformer 핵심 원리: 토큰화부터 Self-Attention, KV Cache까지"
description: "텍스트가 토큰과 벡터가 되고, Self-Attention과 causal mask를 거쳐 다음 토큰으로 생성되는 과정을 수식과 숫자 예제로 정리합니다."
date: 2026-07-30 20:00:00 +0900
tags: [Transformer, LLM, Self-Attention, KV Cache]
reading_time: 12
math: true
image: /assets/images/transformer-fundamentals/pipeline.webp
---

Transformer를 처음 공부하면 토큰화, 임베딩, Self-Attention, causal mask, KV Cache가 서로 떨어진 개념처럼 보인다. 하지만 하나의 생성 파이프라인으로 연결하면 구조가 훨씬 명확해진다.

> **한 문장 요약**  
> 토큰화는 입력을 작은 단위로 나누고, 임베딩과 위치 정보는 각 토큰에 의미와 순서를 부여한다. Self-Attention은 토큰 사이의 관계를 섞고, decoder-only 구조는 왼쪽 문맥만 이용해 다음 토큰을 예측한다. KV Cache는 이미 계산한 정보를 재사용해 이 과정을 실제 서비스 속도로 실행하게 해준다.

![텍스트에서 다음 토큰까지의 Transformer 처리 흐름]({{ '/assets/images/transformer-fundamentals/pipeline.webp' | relative_url }})

## 1. 텍스트가 모델의 입력이 되기까지

모델은 문자열을 그대로 처리하지 못한다. 입력 문장은 먼저 토크나이저를 거쳐 토큰 ID의 시퀀스로 바뀐다.

```text
원문 텍스트 → 토큰화 → 토큰 ID → 임베딩 → 위치 정보 추가 → Transformer 블록
```

### BPE 토크나이제이션

BPE(Byte Pair Encoding)는 자주 함께 등장하는 인접 기호 쌍을 반복적으로 합쳐 subword 어휘집을 만드는 방법이다.

다음과 같은 작은 코퍼스가 있다고 하자.

```text
low
lowest
newer
wider
```

`l`과 `o`가 자주 붙어 있다면 `lo`로 합치고, 이후 `lo`와 `w`가 자주 붙으면 `low`로 합친다. 학습이 끝나면 `lowest`는 `low + est`, `newer`는 `new + er`처럼 분리될 수 있다.

핵심은 **형태소 규칙이 아니라 데이터에서 함께 등장한 빈도**를 바탕으로 경계가 정해진다는 점이다. 그래서 같은 단어도 문맥, 공백, 대소문자, 토크나이저 구현에 따라 다르게 분리될 수 있다.

![BPE와 subword 토큰화에서 자주 만나는 경계 사례]({{ '/assets/images/transformer-fundamentals/subword-edge-cases.webp' | relative_url }})

초보자가 자주 놓치는 사례는 다음과 같다.

- 희귀어는 통째로 사라지지 않고 여러 subword로 분해될 수 있다.
- 숫자와 구두점도 학습된 빈도에 따라 예상 밖의 단위로 나뉠 수 있다.
- byte-level 토크나이저는 이모지와 희귀문자를 UTF-8 바이트 조각으로 표현할 수 있다.
- 한국어도 동일한 원리로 처리되지만, 음절·subword·byte 경계가 형태소 경계와 일치한다는 보장은 없다.

### 임베딩과 위치 정보

토큰 ID는 어휘집의 행 번호일 뿐이므로, 모델은 각 ID를 학습 가능한 벡터로 변환한다.

$$
e_i = E[token_i]
$$

하지만 임베딩만 사용하면 같은 토큰은 위치와 무관하게 같은 벡터를 갖는다. Transformer의 Attention 연산 자체는 입력 순서를 자동으로 알지 못하므로 위치 정보를 추가해야 한다.

$$
h_i^{(0)} = e_i + p_i
$$

여기서 $p_i$는 위치를 나타내는 표현이다. 원 논문은 sinusoidal positional encoding을 사용했지만, 실제 모델은 learned positional embedding이나 RoPE(Rotary Position Embedding)처럼 다른 방식을 사용하기도 한다.

## 2. Self-Attention: 어떤 토큰을 얼마나 참고할 것인가

Self-Attention은 각 토큰이 같은 문장 안의 다른 토큰을 얼마나 참고할지 계산하는 연산이다.

예를 들어 다음 문장을 보자.

> The animal didn't cross the street because **it** was tired.

`it`을 이해하려면 `street`보다 `animal`을 더 강하게 참고해야 한다. Attention은 이런 참고 강도를 모든 토큰 쌍에 대해 계산하고, 그 비율로 정보를 섞어 새로운 표현을 만든다.

### Query, Key, Value

입력 행렬 $X$는 세 개의 선형 변환을 거쳐 Query, Key, Value가 된다.

$$
Q=XW_Q, \qquad K=XW_K, \qquad V=XW_V
$$

직관적으로 해석하면 다음과 같다.

- **Query**: 지금 내가 찾는 정보
- **Key**: 다른 토큰이 나를 찾을 때 비교할 정보
- **Value**: 선택되었을 때 실제로 전달할 내용

Query와 Key의 내적으로 유사도를 구하고, softmax를 통과시켜 합이 1인 가중치로 만든다. 그 가중치로 Value를 섞으면 Attention의 출력이 된다.

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

$\sqrt{d_k}$로 나누는 이유는 차원이 커질수록 내적값의 크기도 커져 softmax가 지나치게 뾰족해지는 현상을 줄이기 위해서다. $M$은 필요할 때 적용하는 mask다.

### 숫자로 계산해 보기

토큰 세 개의 표현과 투영 행렬을 다음처럼 단순화하자.

$$
X=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix},
\qquad W_Q=W_K=W_V=I
$$

그러면 $Q=K=V=X$다. $d_k=2$일 때 점수 행렬은 다음과 같다.

$$
S=\frac{QK^\top}{\sqrt{2}}
=
\begin{bmatrix}
0.707&0&0.707\\
0&0.707&0.707\\
0.707&0.707&1.414
\end{bmatrix}
$$

행별로 softmax를 적용하면 Attention 가중치를 얻는다.

$$
A\approx
\begin{bmatrix}
0.401&0.198&0.401\\
0.198&0.401&0.401\\
0.248&0.248&0.503
\end{bmatrix}
$$

최종 출력은 $Z=AV$다.

$$
Z\approx
\begin{bmatrix}
0.802&0.599\\
0.599&0.802\\
0.751&0.751
\end{bmatrix}
$$

즉, 각 토큰의 새 표현은 자신을 포함한 다른 토큰의 Value를 가중평균한 결과다.

## 3. Causal Mask: 생성 모델이 미래를 보지 못하게 하기

일반적인 Self-Attention에서는 모든 위치가 서로를 볼 수 있다. 하지만 GPT 같은 autoregressive 생성 모델이 학습 중 정답의 미래 토큰을 미리 보면 안 된다.

이를 막기 위해 현재 위치보다 오른쪽에 있는 점수에 $-\infty$를 더한다.

$$
M=
\begin{bmatrix}
0&-\infty&-\infty\\
0&0&-\infty\\
0&0&0
\end{bmatrix}
$$

앞의 예제에 mask를 적용하면 다음과 같은 가중치가 나온다.

$$
A_{causal}\approx
\begin{bmatrix}
1.000&0&0\\
0.330&0.670&0\\
0.248&0.248&0.503
\end{bmatrix}
$$

- 위치 1은 자기 자신만 볼 수 있다.
- 위치 2는 위치 1과 2만 볼 수 있다.
- 위치 3은 위치 1, 2, 3을 모두 볼 수 있다.

이 제약 덕분에 모델은 **지금까지 나온 토큰만 보고 다음 토큰을 예측**한다. 학습할 때는 전체 문장을 병렬로 넣을 수 있지만 mask로 미래 정보만 차단한다. 실제 생성 단계에서는 토큰을 하나씩 순차적으로 만든다.

## 4. Multi-Head Attention

하나의 Attention만 사용하면 모든 관계를 하나의 가중치 공간에 담아야 한다. Multi-Head Attention은 여러 Attention head를 병렬로 계산한 뒤 결과를 합친다.

$$
head_i=\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

$$
\operatorname{MHA}(Q,K,V)=\operatorname{Concat}(head_1,\ldots,head_h)W^O
$$

직관적으로는 한 head가 주어와 동사의 관계에 반응하고, 다른 head가 대명사의 참조나 문장 경계에 반응한다고 생각할 수 있다. 다만 실제 head가 언제나 사람이 이해할 수 있는 역할 하나만 담당하는 것은 아니다.

### Attention의 계산 비용

길이가 $n$인 시퀀스는 모든 토큰 쌍의 점수를 담는 $n\times n$ Attention 행렬을 만든다. 따라서 표준 Self-Attention의 시간 및 Attention 행렬 메모리는 시퀀스 길이에 대해 대체로 $O(n^2)$ 성질을 갖는다.

문맥을 두 배로 늘렸을 때 비용이 단순히 두 배가 되지 않는 이유다.

## 5. 왜 GPT는 Decoder-only 구조를 사용하는가

원래 Transformer는 encoder와 decoder를 함께 사용했다.

- **Encoder-only**: 입력 전체를 양방향으로 이해한다. 분류, 검색, NER 등에 적합하다.
- **Decoder-only**: 왼쪽에서 오른쪽으로 다음 토큰을 생성한다. 채팅, 글쓰기, 코드 생성 등에 적합하다.
- **Encoder-decoder**: encoder가 입력을 이해하고 decoder가 출력을 생성한다. 번역, 요약, text-to-text 변환 등에 적합하다.

![Transformer 계열 아키텍처 비교]({{ '/assets/images/transformer-fundamentals/architecture-comparison.webp' | relative_url }})

GPT 계열의 목표는 간단히 표현하면 다음과 같다.

> 지금까지 나온 토큰들을 보고 다음 토큰을 맞혀라.

모델은 이 과정을 반복해 한 토큰씩 문장을 이어 쓴다. Decoder-only 구조와 causal mask가 자연스럽게 연결되는 이유다.

## 6. KV Cache: 이미 계산한 것은 다시 계산하지 않는다

생성 과정에서 매번 전체 문맥의 Key와 Value를 다시 계산하면 중복이 매우 크다.

예를 들어 `오늘 날씨가`까지 처리한 뒤 `좋다`를 생성한다고 하자. 다음 토큰을 만들 때 과거 네 토큰의 Key와 Value는 변하지 않는다. 따라서 실제 추론 시스템은 각 레이어에서 계산한 과거의 K와 V를 저장해 재사용한다. 이것이 **KV Cache**다.

비유하면 다음과 같다.

- Q는 지금 막 만든 질문이다.
- K와 V는 과거 토큰이 남긴 메모다.
- KV Cache는 그 메모를 레이어마다 쌓아 두는 서랍장이다.

### 무엇을 저장하는가

KV Cache는 보통 다음 정보를 저장한다.

> 각 레이어 × KV head × 지금까지 본 위치에 대한 Key와 Value

대략적인 메모리 사용량은 다음처럼 생각할 수 있다.

$$
2 \times L \times n \times H_{kv} \times d_{head} \times \text{bytes per element}
$$

앞의 2는 K와 V, $L$은 레이어 수, $n$은 문맥 길이, $H_{kv}$는 KV head 수다. Multi-Query Attention이나 Grouped-Query Attention을 쓰는 모델은 Query head보다 KV head 수를 줄여 이 메모리를 절약한다.

### KV Cache가 해결하지 못하는 것

KV Cache는 과거 토큰의 K/V 투영을 다시 계산하는 중복을 없앤다. 그러나 새 토큰의 Query가 과거의 모든 Key를 참고해야 한다는 사실은 사라지지 않는다.

따라서 decoding 단계에서 새 토큰 하나의 Attention 비용은 현재 문맥 길이에 대체로 선형으로 증가한다. $n$개의 토큰을 끝까지 생성하는 전체 과정은 여전히 누적 비용이 커진다.

| 구분 | 장점 | 비용 |
|---|---|---|
| KV Cache 미사용 | 캐시 메모리가 필요 없음 | 과거 K/V 계산을 계속 반복 |
| KV Cache 사용 | 중복 계산을 제거해 생성 속도 향상 | 문맥 길이와 동시 요청 수에 따라 메모리 증가 |

실제 서비스에서는 단순히 모델 연산만이 아니라 메모리 파편화, continuous batching, 요청 간 prefix 공유, 캐시 eviction 전략이 처리량에 큰 영향을 준다.

## 7. 전체 흐름 다시 보기

Transformer 기반 생성 모델의 동작을 순서대로 정리하면 다음과 같다.

1. 원문을 BPE 또는 유사한 방식으로 토큰화한다.
2. 토큰을 정수 ID로 변환한다.
3. ID를 임베딩 벡터로 바꾸고 위치 정보를 반영한다.
4. Self-Attention이 토큰 사이의 관련도를 계산한다.
5. Decoder-only 모델은 causal mask로 미래 토큰을 차단한다.
6. 여러 Transformer 블록을 통과한 표현으로 다음 토큰의 확률을 계산한다.
7. 선택된 토큰을 문맥 뒤에 붙이고 다음 생성을 반복한다.
8. 이때 과거의 K/V는 KV Cache에 저장해 재사용한다.

토큰화부터 KV Cache까지는 서로 독립적인 기능 목록이 아니다. **텍스트를 계산 가능한 표현으로 만들고, 문맥 관계를 학습하며, 그 계산을 실제 시스템에서 효율적으로 반복하기 위한 하나의 연결된 설계**다.

## 참고 자료

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

