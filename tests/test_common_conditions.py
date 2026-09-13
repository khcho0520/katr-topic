"""공통 조건(조건 통일) 적용 검증."""
from __future__ import annotations

from core.models import PRESETS, apply_common_conditions
from core.models.bertopic_model import BERTopicConfig
from core.models.qualit.pipeline import QualITConfig


def _base_configs():
    return [PRESETS[k] for k in ("bertopic", "lda", "nmf", "qualit_paper")]


def test_common_k_applied_to_all():
    configs = apply_common_conditions(_base_configs(), n_topics=10)
    for c in configs:
        assert c.n_topics == 10
        assert "K=10" in c.label
    qualit = next(c for c in configs if isinstance(c, QualITConfig))
    assert qualit.max_topics >= 10


def test_no_override_keeps_originals():
    originals = _base_configs()
    configs = apply_common_conditions(originals)
    for orig, new in zip(originals, configs):
        assert orig.hash() == new.hash()
        assert orig.label == new.label


def test_bertopic_mcs_only_affects_bertopic():
    configs = apply_common_conditions(_base_configs(), bertopic_min_cluster_size=100)
    bertopic = next(c for c in configs if isinstance(c, BERTopicConfig))
    assert bertopic.min_cluster_size == 100
    assert "mcs=100" in bertopic.label
    others = [c for c in configs if not isinstance(c, BERTopicConfig)]
    for c, orig in zip(others, [o for o in _base_configs() if not isinstance(o, BERTopicConfig)]):
        assert c.hash() == orig.hash()


def test_qualit_llm_docs_override():
    configs = apply_common_conditions(_base_configs(), qualit_max_llm_docs=1000)
    qualit = next(c for c in configs if isinstance(c, QualITConfig))
    assert qualit.max_llm_docs == 1000
    assert "llm≤1000" in qualit.label
    # 같은 값이면 접미사 없음
    same = apply_common_conditions(_base_configs(), qualit_max_llm_docs=500)
    qualit_same = next(c for c in same if isinstance(c, QualITConfig))
    assert qualit_same.label == "QualIT (논문 재현)"


def test_override_changes_hash_and_run_id():
    from core.experiments.runner import make_run_id

    base = _base_configs()[0]
    fixed = apply_common_conditions([base], n_topics=10)[0]
    assert base.hash() != fixed.hash()
    assert make_run_id(base, 42) != make_run_id(fixed, 42)
