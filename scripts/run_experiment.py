"""헤드리스 실험 실행 CLI (Streamlit 없이 대규모 그리드 실행용).

예:
  .venv/bin/python scripts/run_experiment.py \\
      --corpus-hash <해시> --models bertopic,lda,nmf,qualit_paper,qualit_improved \\
      --seeds 42,43,44,45,46 --experiment-id exp_full --judge
저장된 코퍼스 목록: --list-corpora
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data.corpus_store import list_corpora, load_corpus  # noqa: E402
from core.data.prepare import get_corpus_embeddings  # noqa: E402
from core.experiments.export import to_csv, to_latex_table, to_markdown_table  # noqa: E402
from core.experiments.runner import run_experiment  # noqa: E402
from core.models import PRESETS, apply_common_conditions  # noqa: E402
from core.utils.progress import make_console_progress  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-corpora", action="store_true")
    parser.add_argument("--corpus-hash")
    parser.add_argument("--models", default="bertopic,lda,nmf,qualit_paper")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--experiment-id", required=False)
    parser.add_argument("--judge", action="store_true", help="LLM-as-judge 평가 포함")
    parser.add_argument("--n-topics", type=int, default=0,
                        help="공통 토픽 수 K — 모든 모델에 동일 K 고정 (0=모델별 자동)")
    parser.add_argument("--bertopic-mcs", type=int, default=0,
                        help="BERTopic HDBSCAN min_cluster_size (0=자동 n/20)")
    parser.add_argument("--qualit-max-llm-docs", type=int, default=-1,
                        help="QualIT LLM 대표 문서 상한 (-1=프리셋 기본)")
    parser.add_argument("--auto-conditions", action="store_true",
                        help="데이터 특성 기반 공통 조건 자동 제안 적용 "
                             "(명시한 --n-topics 등이 있으면 그 값이 우선)")
    args = parser.parse_args()

    if args.list_corpora:
        for m in list_corpora():
            print(f"{m['content_hash']}  {m['name']:<20} {m['regime']:<10} {m['n_docs']}건")
        return 0

    if not args.corpus_hash:
        parser.error("--corpus-hash가 필요합니다 (--list-corpora로 확인)")

    unknown = [k for k in args.models.split(",") if k not in PRESETS]
    if unknown:
        parser.error(f"알 수 없는 프리셋: {unknown} (가능: {list(PRESETS)})")

    corpus = load_corpus(args.corpus_hash)
    embeddings = get_corpus_embeddings(corpus)
    if args.auto_conditions:
        from core.experiments.recommend import recommend_common_conditions

        reco = recommend_common_conditions(corpus, embeddings)
        print("자동 제안 근거:")
        for line in reco.rationale:
            print(f"  - {line}")
        if args.n_topics == 0:
            args.n_topics = reco.n_topics
        if args.bertopic_mcs == 0:
            args.bertopic_mcs = reco.bertopic_min_cluster_size
        if args.qualit_max_llm_docs < 0:
            args.qualit_max_llm_docs = reco.qualit_max_llm_docs
    configs = apply_common_conditions(
        [PRESETS[k] for k in args.models.split(",")],
        n_topics=args.n_topics,
        bertopic_min_cluster_size=args.bertopic_mcs,
        qualit_max_llm_docs=None if args.qualit_max_llm_docs < 0 else args.qualit_max_llm_docs,
    )
    print("실행 설정:", ", ".join(c.label for c in configs))
    seeds = [int(s) for s in args.seeds.split(",")]
    experiment_id = args.experiment_id or f"exp_{corpus.name}_{corpus.content_hash[:8]}"

    records = list(
        run_experiment(
            corpus, embeddings, configs, seeds, experiment_id,
            judge=args.judge, progress=make_console_progress(),
        )
    )
    print()
    print(to_markdown_table(records))
    out_dir = Path("runs") / experiment_id
    (out_dir / "summary.csv").write_text(to_csv(records), encoding="utf-8")
    (out_dir / "table.tex").write_text(to_latex_table(records), encoding="utf-8")
    print(f"\n요약 저장: {out_dir}/summary.csv, table.tex")
    n_failed = sum(1 for r in records if r.status == "failed")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
