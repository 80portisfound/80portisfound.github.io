---
layout: post
title: "FastAPI로 CRUD API 만들기: Pydantic, SQLAlchemy, JWT까지"
description: "간단한 엔드포인트에서 시작해 요청·응답 검증, SQLAlchemy 2.0 CRUD, 의존성 주입, JWT 인증과 작성자 권한까지 FastAPI 프로젝트를 단계별로 구성합니다."
date: 2026-08-17 20:00:00 +0900
tags: [FastAPI, SQLAlchemy, Pydantic, JWT]
category: development
category_label: 개발
reading_time: 18
---

FastAPI 입문 예제는 `Hello World`에서 시작하지만 실제 API에는 데이터 검증, 데이터베이스 세션, CRUD, 인증과 권한 검사가 함께 필요하다. 기능이 늘어날수록 모든 코드를 `main.py`에 두는 방식도 금방 한계에 도달한다.

이 글에서는 질문과 답변을 저장하는 작은 API를 다음 순서로 확장한다.

```text
Route → Pydantic Schema → APIRouter
      → SQLAlchemy Session → CRUD
      → JWT Authentication → Authorization
```

원문의 단계별 학습 흐름은 유지하되, 반복되는 전체 파일 대신 각 계층의 책임과 핵심 코드에 집중한다. 예제는 Pydantic 2와 SQLAlchemy 2 스타일을 기준으로 한다.

## 1. 가장 작은 FastAPI 애플리케이션

필요한 패키지를 설치한다.

```bash
pip install fastapi "uvicorn[standard]" sqlalchemy \
  pydantic-settings pwdlib PyJWT python-multipart
```

`main.py`를 만든다.

```python
from fastapi import FastAPI

app = FastAPI(title="Q&A API")


@app.get("/")
def root():
    return {"message": "FastAPI study started"}


@app.get("/health")
def health():
    return {"status": "ok"}
```

개발 서버를 실행한다.

```bash
uvicorn main:app --reload
```

`main:app`은 `main.py` 모듈의 `app` 객체를 뜻한다. `--reload`는 코드 변경 시 서버를 다시 시작하는 개발용 옵션이다. 서버가 실행되면 `/docs`에서 Swagger UI, `/redoc`에서 ReDoc 문서를 확인할 수 있다.

## 2. Path Parameter와 Query Parameter

URL 경로 자체에 포함된 값은 Path Parameter다.

```python
@app.get("/questions/{question_id}")
def get_question(question_id: int):
    return {
        "question_id": question_id,
        "subject": f"Question {question_id}",
    }
```

`question_id: int`라는 타입 힌트 덕분에 FastAPI는 문자열 입력을 정수로 변환하고, 변환할 수 없으면 자동으로 `422 Unprocessable Entity` 응답을 만든다.

목록의 검색 범위를 정하는 값은 Query Parameter로 표현할 수 있다.

```python
from fastapi import Query


@app.get("/questions")
def get_questions(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=10, ge=1, le=100),
):
    return {"skip": skip, "limit": limit}
```

`ge`는 이상, `le`는 이하 조건이다. 검증 규칙은 문서에도 자동으로 반영된다.

## 3. Pydantic으로 요청과 응답을 분리하기

POST 요청 본문은 Pydantic 모델로 검증한다.

```python
from pydantic import BaseModel, ConfigDict, Field


class QuestionCreate(BaseModel):
    subject: str = Field(min_length=2, max_length=100)
    content: str = Field(min_length=1)


class QuestionUpdate(BaseModel):
    subject: str | None = Field(
        default=None,
        min_length=2,
        max_length=100,
    )
    content: str | None = Field(default=None, min_length=1)


class QuestionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    subject: str
    content: str
```

요청과 응답 스키마를 분리하는 이유는 데이터의 방향이 다르기 때문이다.

- `QuestionCreate`: 클라이언트가 보내야 하는 값
- `QuestionUpdate`: 부분 수정에서 선택적으로 보내는 값
- `QuestionResponse`: 서버가 ID를 포함해 공개할 값

`from_attributes=True`는 SQLAlchemy ORM 객체의 속성에서 응답 데이터를 읽도록 허용한다. ORM 모델을 그대로 반환하면서 이 설정을 빠뜨리면 응답 검증에서 문제가 생길 수 있다.

```python
@app.post(
    "/questions",
    response_model=QuestionResponse,
    status_code=201,
)
def create_question(question: QuestionCreate):
    return {"id": 1, **question.model_dump()}
```

`response_model`은 문서용 타입 표시만 하는 것이 아니다. 실제 응답을 검증하고 선언된 필드만 직렬화하는 경계다. 비밀번호 해시 같은 내부 필드가 응답으로 새지 않도록 별도 응답 스키마를 사용해야 한다.

## 4. APIRouter로 기능을 분리하기

라우트가 늘어나면 `main.py`에는 앱 조립만 남기고 도메인별 Router를 분리한다.

```text
app/
├── main.py
├── database.py
├── models.py
├── auth.py
├── security.py
├── routers/
│   ├── question.py
│   ├── answer.py
│   └── user.py
├── schemas/
│   ├── question.py
│   ├── answer.py
│   └── user.py
└── crud/
    ├── question.py
    ├── answer.py
    └── user.py
```

```python
# routers/question.py
from fastapi import APIRouter

router = APIRouter(
    prefix="/questions",
    tags=["Questions"],
)


@router.get("/{question_id}")
def get_question(question_id: int):
    return {"question_id": question_id}
```

```python
# main.py
from fastapi import FastAPI
from routers import question

app = FastAPI(title="Q&A API")
app.include_router(question.router)
```

`prefix`를 Router에 두면 각 함수에서 `/questions`를 반복하지 않아도 된다. `tags`는 API 문서에서 엔드포인트를 그룹화한다.

## 5. SQLAlchemy Engine과 Session

SQLite를 사용하는 동기식 SQLAlchemy 설정은 다음과 같다.

```python
# database.py
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = "sqlite:///./app.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

Engine은 애플리케이션과 DB 사이의 연결 자원을 관리하고, Session은 한 작업 단위에서 조회·추가·수정·삭제를 수행하는 공간이다. `sessionmaker`는 같은 설정의 Session을 만드는 Factory다.

`get_db()`는 요청마다 Session을 제공하고 요청이 끝나면 닫는다. FastAPI의 `Depends`와 Generator를 결합하면 이 생명주기를 일관되게 관리할 수 있다.

> 이 예제는 동기식 Session을 사용하므로 엔드포인트도 일반 `def`로 작성한다. 동기 DB 호출을 `async def` 안에서 직접 실행하면 Event Loop를 막을 수 있다. 비동기 구성이 필요하다면 SQLAlchemy의 `AsyncSession`과 비동기 Driver를 함께 사용해야 한다.

## 6. ORM 모델과 관계 정의

질문과 사용자의 관계를 포함한 ORM 모델을 정의한다.

```python
# models.py
from __future__ import annotations

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        index=True,
    )
    password_hash: Mapped[str] = mapped_column(String(255))

    questions: Mapped[list[Question]] = relationship(
        back_populates="author",
    )


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(primary_key=True)
    subject: Mapped[str] = mapped_column(String(200), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    author_id: Mapped[int] = mapped_column(ForeignKey("users.id"))

    author: Mapped[User] = relationship(back_populates="questions")
    answers: Mapped[list[Answer]] = relationship(
        back_populates="question",
        cascade="all, delete-orphan",
    )


class Answer(Base):
    __tablename__ = "answers"

    id: Mapped[int] = mapped_column(primary_key=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    question_id: Mapped[int] = mapped_column(
        ForeignKey("questions.id"),
    )
    author_id: Mapped[int] = mapped_column(ForeignKey("users.id"))

    question: Mapped[Question] = relationship(back_populates="answers")
```

`QuestionCreate`는 HTTP 요청 구조이고 `Question`은 DB Table 구조다. 이름이 비슷해도 역할은 다르다.

학습 단계에서는 다음 코드로 Table을 만들 수 있다.

```python
Base.metadata.create_all(bind=engine)
```

하지만 운영 환경의 Schema 변경에는 `create_all()`만으로 부족하다. Column 추가와 데이터 변환 이력을 관리하려면 Alembic 같은 Migration 도구를 사용해야 한다.

## 7. Depends로 DB Session 주입하기

```python
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
import models


@router.get(
    "/{question_id}",
    response_model=QuestionResponse,
)
def get_question(
    question_id: int,
    db: Session = Depends(get_db),
):
    question = db.get(models.Question, question_id)
    if question is None:
        raise HTTPException(
            status_code=404,
            detail="Question not found",
        )
    return question
```

라우트 함수는 Session을 직접 만들거나 닫지 않는다. `Depends(get_db)`가 요청 단위 Session을 주입한다. 같은 패턴을 인증, 권한, 설정 로딩에도 재사용할 수 있다.

## 8. CRUD를 별도 계층으로 옮기기

Router는 HTTP 입력과 상태 코드를 담당하고 CRUD 계층은 DB Query와 상태 변경을 담당하게 분리한다.

```python
# crud/question.py
from sqlalchemy import select
from sqlalchemy.orm import Session

import models
from schemas.question import QuestionCreate, QuestionUpdate


def get_question(db: Session, question_id: int):
    return db.get(models.Question, question_id)


def get_questions(
    db: Session,
    *,
    skip: int = 0,
    limit: int = 10,
    keyword: str | None = None,
):
    statement = select(models.Question).order_by(
        models.Question.id.desc()
    )

    if keyword:
        statement = statement.where(
            models.Question.subject.contains(keyword)
        )

    statement = statement.offset(skip).limit(limit)
    return list(db.scalars(statement).all())


def create_question(
    db: Session,
    data: QuestionCreate,
    author_id: int,
):
    question = models.Question(
        **data.model_dump(),
        author_id=author_id,
    )
    db.add(question)
    db.commit()
    db.refresh(question)
    return question
```

목록 API에는 명시적인 `order_by`가 필요하다. 정렬 기준 없이 Offset Pagination을 사용하면 요청 사이에 결과 순서가 안정적이라는 보장이 없다. 데이터가 커지면 Offset 대신 Cursor Pagination을 검토할 수 있다.

DB 오류가 발생할 수 있는 서비스에서는 Transaction 경계를 명확히 하고 예외 시 `rollback()`을 수행해야 한다. 규모가 커지면 CRUD 함수마다 Commit할지, Service 계층에서 하나의 Transaction으로 묶을지 정책을 정한다.

## 9. PUT, PATCH, DELETE의 차이

PUT은 Resource의 전체 표현을 교체하는 의미로, PATCH는 일부 필드만 수정하는 의미로 사용하는 경우가 일반적이다.

```python
def patch_question(
    db: Session,
    question: models.Question,
    data: QuestionUpdate,
):
    updates = data.model_dump(exclude_unset=True)
    for field, value in updates.items():
        setattr(question, field, value)

    db.commit()
    db.refresh(question)
    return question
```

`exclude_unset=True`는 클라이언트가 보내지 않은 필드를 구분한다. 단순히 `value is not None`만 검사하면 “필드를 보내지 않음”과 “명시적으로 null을 보냄”을 구분할 수 없다. null을 허용할지까지 API 정책으로 정해야 한다.

DELETE 성공 시 `204 No Content`를 사용한다면 응답 Body를 만들지 않는다.

```python
from fastapi import Response, status


@router.delete(
    "/{question_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_question(
    question_id: int,
    db: Session = Depends(get_db),
):
    question = question_crud.get_question(db, question_id)
    if question is None:
        raise HTTPException(404, "Question not found")
    question_crud.delete_question(db, question)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
```

## 10. 비밀번호는 해시로 저장하기

회원가입 요청과 공개 응답을 분리한다.

```python
# schemas/user.py
from pydantic import BaseModel, ConfigDict, Field


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=128)


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
```

비밀번호는 평문이나 복호화 가능한 형태로 저장하지 않는다.

```python
# security.py
from pwdlib import PasswordHash

password_hash = PasswordHash.recommended()


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return password_hash.verify(password, hashed)
```

응답 스키마에는 `password`와 `password_hash`를 포함하지 않는다. 로그와 오류 메시지에도 평문 비밀번호가 남지 않도록 주의한다.

## 11. JWT Access Token 발급하기

설정값은 코드에 고정하지 않고 환경 변수에서 읽는다.

```python
# settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30


settings = Settings()
```

```python
# security.py
from datetime import datetime, timedelta, timezone

import jwt

from settings import settings


def create_access_token(user_id: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(
            minutes=settings.access_token_expire_minutes
        ),
    }
    return jwt.encode(
        payload,
        settings.secret_key,
        algorithm=settings.jwt_algorithm,
    )
```

`sub`에는 변경될 수 있는 Username보다 안정적인 User ID를 문자열로 넣는 편이 관리하기 쉽다. JWT Payload는 암호화되는 것이 아니라 서명되는 것이므로 비밀번호나 민감 정보를 넣으면 안 된다.

OAuth2 Password Form을 사용하는 로그인 엔드포인트는 다음과 같다.

```python
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm


@router.post("/login", response_model=TokenResponse)
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = user_crud.get_by_username(db, form.username)
    if user is None or not verify_password(
        form.password,
        user.password_hash,
    ):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "access_token": create_access_token(user.id),
        "token_type": "bearer",
    }
```

## 12. 현재 사용자 인증 의존성

```python
# auth.py
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jwt import InvalidTokenError
import jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    credentials_error = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        user_id = int(payload["sub"])
    except (InvalidTokenError, KeyError, ValueError):
        raise credentials_error

    user = db.get(models.User, user_id)
    if user is None:
        raise credentials_error
    return user
```

이 Dependency를 사용하면 보호된 Endpoint마다 토큰 해석 코드를 반복하지 않아도 된다.

```python
@router.get("/users/me", response_model=UserResponse)
def get_me(
    current_user: models.User = Depends(get_current_user),
):
    return current_user
```

## 13. 인증과 권한은 다르다

인증은 사용자가 누구인지 확인하는 과정이고, 권한 검사는 그 사용자가 특정 Resource를 수정할 수 있는지 판단하는 과정이다.

```python
@router.put(
    "/{question_id}",
    response_model=QuestionResponse,
)
def update_question(
    question_id: int,
    data: QuestionCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    question = question_crud.get_question(db, question_id)
    if question is None:
        raise HTTPException(404, "Question not found")

    if question.author_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to update this question",
        )

    return question_crud.update_question(db, question, data)
```

- 토큰이 없거나 유효하지 않으면 `401 Unauthorized`
- 로그인은 했지만 해당 작업 권한이 없으면 `403 Forbidden`
- Resource가 존재하지 않으면 `404 Not Found`

Resource 존재 여부를 권한 없는 사용자에게 숨기기 위해 일관되게 404를 반환하는 정책도 있다. 어떤 방식을 선택하든 API 전체에서 일관돼야 한다.

## 14. 질문과 답변 API의 전체 요청 흐름

```text
HTTP Request
    ↓
FastAPI Router
    ↓
Pydantic Request Validation
    ↓
Depends(get_db) + Depends(get_current_user)
    ↓
Authentication·Authorization
    ↓
CRUD / Service Logic
    ↓
SQLAlchemy Session·ORM
    ↓
SQLite 또는 운영 Database
    ↓
Pydantic Response Validation
    ↓
HTTP Response
```

이 구조에서 각 계층의 책임은 비교적 명확하다.

| 계층 | 책임 |
| --- | --- |
| Router | HTTP 입력, 상태 코드, Dependency 조합 |
| Schema | 요청·응답 데이터 계약과 검증 |
| Auth | 토큰 검증과 현재 사용자 조회 |
| CRUD/Service | Query, 상태 변경, 비즈니스 규칙 |
| ORM Model | Table과 관계 정의 |
| Database | Engine과 Session 생명주기 |

프로젝트가 작을 때는 CRUD와 Service를 하나로 시작해도 된다. 여러 DB 변경을 하나의 Transaction으로 묶거나 외부 API와 정책 로직이 섞이기 시작하면 Service 계층을 분리하는 편이 좋다.

## 15. 운영 전에 확인할 것

학습용 API에서 운영 서비스로 넘어갈 때는 다음 항목을 추가로 확인해야 한다.

- Alembic Migration과 배포 순서
- 환경별 DB URL과 Secret 관리
- CORS 허용 Origin 최소화
- 요청 크기, Rate Limit, Timeout
- Transaction과 Rollback 정책
- 구조화된 로그와 Request ID
- 테스트 DB를 사용하는 API·권한 테스트
- Access Token 만료와 Refresh Token 전략
- HTTPS, Cookie를 쓸 경우 Secure·HttpOnly·SameSite 설정
- N+1 Query와 Pagination 성능

FastAPI는 입력 검증과 문서 생성, Dependency 조합을 간결하게 만든다. 하지만 데이터 무결성, Transaction, 인증과 권한은 Framework가 자동으로 완성해 주는 영역이 아니다.

결론적으로 FastAPI 프로젝트를 안정적으로 키우는 핵심은 Endpoint 수가 아니라 **HTTP 계약, DB 작업, 인증과 권한의 경계를 분리하는 것**이다. 작은 Router에서 시작해 Schema, Session, CRUD, Auth를 하나씩 분리하면 코드의 성장 과정과 각 계층이 필요한 이유를 함께 이해할 수 있다.

## 더 읽어보기

- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [SQLAlchemy 2.0 ORM Quick Start](https://docs.sqlalchemy.org/en/20/orm/quickstart.html)
- [Pydantic Models](https://docs.pydantic.dev/latest/concepts/models/)
- [Alembic Tutorial](https://alembic.sqlalchemy.org/en/latest/tutorial.html)
