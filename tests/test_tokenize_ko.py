"""Kiwi 토크나이저: 명사 추출, 불용어 제거, 키워드 정규화."""
from core.data.tokenize_ko import normalize_keyword, tokenize


def test_tokenize_extracts_nouns():
    tokens = tokenize("정부가 인공지능 정책을 발표했다")
    assert "정책" in tokens
    assert any("인공" in t or "지능" in t or t == "인공지능" for t in tokens)
    # 조사/동사는 제외
    assert "발표했다" not in tokens


def test_tokenize_removes_stopwords_and_short():
    tokens = tokenize("이번 발표 관련 내용")   # 모두 불용어 사전에 있음
    assert "이번" not in tokens
    assert "관련" not in tokens


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize("   ") == []


def test_normalize_keyword_phrase():
    tokens = normalize_keyword("저출산 대응 정책")
    assert "정책" in tokens
    assert all(len(t) >= 2 for t in tokens)


def test_english_lowercased():
    tokens = tokenize("정부 AI 규제 발표")
    assert "ai" in tokens
