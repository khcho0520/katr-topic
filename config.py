"""전역 설정: 경로, Ollama 호스트, 기본 모델명."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "cache"
RUNS_DIR = PROJECT_ROOT / "runs"
CORPORA_DIR = CACHE_DIR / "corpora"
DATA_DIR = PROJECT_ROOT / "data"
STOPWORDS_PATH = DATA_DIR / "stopwords_ko.txt"

for _d in (CACHE_DIR, RUNS_DIR, CORPORA_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def get_ollama_hosts() -> list[str]:
    """OLLAMA_HOSTS 환경변수(쉼표 구분, 첫 번째가 우선)를 파싱한다. 미설정 시 로컬."""
    raw = os.environ.get("OLLAMA_HOSTS", "").strip()
    hosts: list[str] = []
    for part in raw.split(","):
        h = part.strip().rstrip("/")
        if not h:
            continue
        if "://" not in h:
            h = f"http://{h}"
        hosts.append(h)
    return hosts or ["http://localhost:11434"]


# 기본 모델 (로컬 Ollama에 설치된 모델 기준)
DEFAULT_EMBED_MODEL = os.environ.get("TOPICNEW_EMBED_MODEL", "bge-m3:latest")
DEFAULT_LLM_MODEL = os.environ.get("TOPICNEW_LLM_MODEL", "qwen3.5:9b")
DEFAULT_JUDGE_MODEL = os.environ.get("TOPICNEW_JUDGE_MODEL", "qwen3.5:27b")

# LLM 호출 동시성 (로컬 Ollama 기준)
LLM_WORKERS = int(os.environ.get("TOPICNEW_LLM_WORKERS", "4"))

# 임베딩 배치 설정
EMBED_BATCH_SIZE = int(os.environ.get("TOPICNEW_EMBED_BATCH", "16"))
EMBED_TIMEOUT = int(os.environ.get("TOPICNEW_EMBED_TIMEOUT", "120"))

# LLM 타임아웃 (대형 모델은 길게)
LLM_TIMEOUT_LIGHT = 180
LLM_TIMEOUT_HEAVY = 600
HEAVY_MODEL_MARKERS = ("27b", "30b", "32b", "34b", "35b", "70b", "72b")

# 캐시 파일
EMBED_CACHE_PATH = CACHE_DIR / "embeddings.sqlite"
LLM_CACHE_PATH = CACHE_DIR / "llm_cache.sqlite"

# 프롬프트 버전: 프롬프트 문구를 바꾸면 올려서 캐시를 무효화한다
PROMPT_VERSION = "v1"
