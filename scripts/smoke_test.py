"""스모크 테스트: 샘플 헤드라인 → 2시드 × 4모델 전체 파이프라인 완주 확인.

실행: .venv/bin/python scripts/smoke_test.py [--seeds 42,43] [--models bertopic,lda,nmf,qualit_paper]
2회 연속 실행 시 두 번째는 LLM 신규 호출이 0건이어야 한다 (캐시 검증).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR  # noqa: E402
from core.data.bigkinds import load_bigkinds  # noqa: E402
from core.data.prepare import prepare_corpus  # noqa: E402
from core.experiments.export import to_markdown_table  # noqa: E402
from core.experiments.runner import run_experiment  # noqa: E402
from core.llm.ollama_client import get_default_client  # noqa: E402
from core.models import PRESETS  # noqa: E402
from core.utils.progress import make_console_progress  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--models", default="bertopic,lda,nmf,qualit_paper")
    parser.add_argument("--experiment-id", default="smoke_test")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    configs = [PRESETS[k] for k in args.models.split(",")]

    llm = get_default_client()
    if not llm.ping():
        print("❌ Ollama에 연결할 수 없습니다. `ollama serve`를 확인하세요.")
        return 1

    print("1) 샘플 코퍼스 준비 (토큰화 + 임베딩)")
    sample = DATA_DIR / "samples" / "sample_bigkinds.csv"
    corpus = load_bigkinds(sample, filename=sample.name, name="smoke_sample")
    corpus, embeddings = prepare_corpus(corpus, progress=make_console_progress("  "))
    print(f"   코퍼스: {len(corpus)}건, 임베딩 {embeddings.shape}")

    print(f"2) 실험 실행: {len(configs)}개 설정 × {len(seeds)}개 시드")
    calls_before = llm.call_count
    records = list(
        run_experiment(
            corpus, embeddings, configs, seeds,
            experiment_id=args.experiment_id,
            progress=make_console_progress("  "),
        )
    )
    print(f"   신규 LLM HTTP 호출: {llm.call_count - calls_before}건")

    print("3) 검증")
    failures = []
    for r in records:
        if r.status != "done":
            failures.append(f"{r.run_id}: 실행 실패 — {r.error[:200]}")
            continue
        n_topics = r.metrics.get("n_topics_found", 0)
        if n_topics < 2:
            failures.append(f"{r.run_id}: 토픽 수 {n_topics} < 2")
        for key, value in r.metrics.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                failures.append(f"{r.run_id}: 지표 {key}={value} 비정상")

    print()
    print(to_markdown_table(records))
    print()
    if failures:
        print("❌ 스모크 실패:")
        for f in failures:
            print(f"   - {f}")
        return 1
    print(f"✅ 스모크 통과: {len(records)} runs 완료, 전 지표 유한값")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
