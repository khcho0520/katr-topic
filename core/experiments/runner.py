"""실험 러너: (설정 × 시드) 그리드 실행 + 재개(resume).

- 완료된 run_id는 스킵하고 캐시된 기록을 반환 → 중단 후 재실행 시 빠르게 이어감
- LLM 단계는 디스크 캐시 덕분에 시드 간 재사용됨 (클러스터링만 재계산)
"""
from __future__ import annotations

import logging
import traceback
from datetime import datetime
from typing import Iterator, List, Optional

import numpy as np

from core.experiments import store
from core.metrics.suite import evaluate
from core.models.base import ModelConfig, create_model
from core.types import Corpus, RunRecord, TopicResult
from core.utils.progress import ProgressCallback

logger = logging.getLogger(__name__)


def make_run_id(config: ModelConfig, seed: int) -> str:
    return f"{config.name}_{config.hash()}_s{seed}"


def run_single(
    corpus: Corpus,
    embeddings: np.ndarray,
    config: ModelConfig,
    seed: int,
    experiment_id: str,
    judge: bool = False,
    judge_model: Optional[str] = None,
    progress: Optional[ProgressCallback] = None,
) -> RunRecord:
    """1개 (설정, 시드) 실행 → 평가 → RunRecord 생성·저장."""
    run_id = make_run_id(config, seed)
    try:
        model = create_model(config)
        result: TopicResult = model.fit(corpus, embeddings, seed, progress=progress)
        report = evaluate(result, corpus, embeddings, judge=judge, judge_model=judge_model)
        topics = {
            str(tid): {
                "label": result.topic_labels.get(tid, ""),
                "keywords": (result.topic_keywords.get(tid) or [])[:10],
                "size": int(np.sum(result.assignments == tid)),
            }
            for tid in result.topic_ids
        }
        record = RunRecord(
            run_id=run_id,
            experiment_id=experiment_id,
            corpus_hash=corpus.content_hash,
            corpus_name=corpus.name,
            regime=corpus.regime,
            model_name=config.name,
            config=config.to_dict(),
            config_hash=config.hash(),
            seed=seed,
            metrics={k: float(v) for k, v in report.scalars.items()},
            topics=topics,
            created_at=datetime.now().isoformat(timespec="seconds"),
            status="done",
            assignments=[int(a) for a in result.assignments],
        )
    except Exception as e:
        logger.error("실행 실패 %s: %s", run_id, e, exc_info=True)
        record = RunRecord(
            run_id=run_id,
            experiment_id=experiment_id,
            corpus_hash=corpus.content_hash,
            corpus_name=corpus.name,
            regime=corpus.regime,
            model_name=config.name,
            config=config.to_dict(),
            config_hash=config.hash(),
            seed=seed,
            metrics={},
            topics={},
            created_at=datetime.now().isoformat(timespec="seconds"),
            status="failed",
            error=f"{e}\n{traceback.format_exc()[-1000:]}",
        )
    store.save_run(record)
    return record


def run_experiment(
    corpus: Corpus,
    embeddings: np.ndarray,
    configs: List[ModelConfig],
    seeds: List[int],
    experiment_id: str,
    judge: bool = False,
    judge_model: Optional[str] = None,
    resume: bool = True,
    progress: Optional[ProgressCallback] = None,
) -> Iterator[RunRecord]:
    """그리드 실행. 완료된 run은 스킵(재개). 실행 순서: 설정별 × 시드별."""
    grid = [(config, seed) for config in configs for seed in seeds]
    total = len(grid)
    for i, (config, seed) in enumerate(grid):
        run_id = make_run_id(config, seed)
        label = f"[{i + 1}/{total}] {config.label} (seed={seed})"
        if resume and store.run_exists(experiment_id, run_id):
            cached = store.load_run(experiment_id, run_id)
            if cached is not None and cached.status == "done":
                if progress:
                    progress((i + 1) / total, f"{label} — 완료됨(스킵)")
                yield cached
                continue
        if progress:
            progress(i / total, f"{label} 실행 중")
        record = run_single(
            corpus, embeddings, config, seed, experiment_id,
            judge=judge, judge_model=judge_model,
        )
        if progress:
            progress((i + 1) / total, f"{label} — {record.status}")
        yield record
