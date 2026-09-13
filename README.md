# KATR — Keyphrase-Anchored Topic Refinement

Code and result files for the paper *Keyphrase-Anchored Topic Refinement for Full-Coverage and Stable Topic Modeling of Short Texts* (submitted to the International Journal of Advanced Smart Convergence).

KATR clusters document embeddings with K-means so that every document gets a topic, and uses a local LLM to build a keyphrase anchor for each topic and to write a one-sentence description. The weight of the anchor in the assignment is a parameter `alpha`; `alpha = 1` switches the anchor off and is the built-in control. The paper compares KATR with LDA, NMF, BERTopic, BERTopic-full, and a reproduction of QualIT on 60,363 Korean news headlines with five random seeds.

## What is in this repository

| Path | Content |
|---|---|
| `core/models/katr_model.py` | KATR (four steps; see the module docstring and Appendix A of the paper) |
| `core/models/qualit/` | QualIT reproduction and its ablation variants |
| `core/models/bertopic_model.py`, `lda_model.py`, `nmf_model.py` | Baselines on the same shared embeddings and tokens |
| `core/metrics/` | Intrinsic, extrinsic, and stability metrics (definitions in `docs/METRICS.md`) |
| `core/experiments/` | Batch runner, paired tests (t-test, Wilcoxon, Cohen's d, Holm) |
| `scripts/run_experiment.py` | Headless CLI for the seed × setting grid |
| `results/` | Metric values behind Tables 1–5 of the paper (no article text; see below) |
| `requirements-lock.txt` | Exact package versions used for the paper |

The prompts used for keyphrase extraction and topic description are in `core/models/qualit/keyphrase.py` (`_build_prompt`) and `core/models/qualit/labeling.py` (`distill_theme`). They are reproduced with an English translation in Appendix A of the paper.

## Results files

`results/` contains, for each of the ten settings in the paper and each seed (42–46):

- `results/runs/<config>_s<seed>.json` — the setting, the metric values, and the topics (LLM description, top-10 c-TF-IDF words, size). Document assignments and headline texts are not included.
- `results/summary_paper_tables.csv` — means and standard deviations over the five seeds (Tables 1, 2, 3, 5).
- `results/per_seed_metrics.json` — per-seed values used for the paired tests in Table 4.
- `results/stability_seed_pairs.json` — ARI between the ten seed pairs of each setting (Table 3).
- `results/paper_settings.json` — mapping from the labels used in the paper to the run files.

## Data

The headlines were collected from BigKinds (Korea Press Foundation, https://www.bigkinds.or.kr) and cannot be redistributed under its terms of use. The paper (Section 3.1) gives the collection conditions: five national daily newspapers (Chosun Ilbo, JoongAng Ilbo, Dong-A Ilbo, Hankyoreh, Kyunghyang Shinmun), January 1 to March 31, 2026, articles whose first-listed category is politics, society, economy, or international, editorial cartoons excluded, duplicates not removed.

To run the pipeline on your own BigKinds export (Excel or CSV with the title, date, press, and `통합 분류1` columns), use page 1 of the Streamlit app or `core/data/bigkinds.py`. Note that `load_bigkinds` removes duplicate headlines by default (`dedup=True`); the paper corpus was built with `dedup=False`.

## Setup

Tested on macOS 15 (Apple M4 Pro, 64 GB) with Python 3.11.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt

# Ollama models (all computation is local)
ollama pull bge-m3:latest
ollama pull qwen3.5:9b        # Q4_K_M
```

## Reproducing the paper tables

1. Prepare the corpus (page 1 of the app, or the MCPanal import used in the paper). This tokenizes the headlines with Kiwi and computes BGE-M3 embeddings once; both are cached under `cache/`.
2. List the prepared corpora and run the grid:

```bash
.venv/bin/python scripts/run_experiment.py --list-corpora
.venv/bin/python scripts/run_experiment.py \
    --corpus-hash <hash> \
    --models lda,nmf,bertopic,bertopic_full,qualit_paper,qualit_joint,katr_alpha03,katr,katr_noanchor,katr_core \
    --seeds 42,43,44,45,46 \
    --n-topics 20 --bertopic-mcs 150 --qualit-max-llm-docs 1000
```

`--n-topics 20`, `--bertopic-mcs 150`, and `--qualit-max-llm-docs 1000` are the common conditions of Section 3.3. Preset names map to the paper as follows: `qualit_paper` = QualIT, `qualit_joint` = QualIT-combined, `katr_alpha03` = KATR (α = 0.3), `katr` = KATR (α = 0.6), `katr_noanchor` = KATR (α = 1), `katr_core` = KATR-core.

3. Page 4 of the app (or `core/experiments/stats.py`) produces the mean ± SD tables, the paired tests with Holm adjustment, and the seed-pair stability table.

LLM outputs are cached in `cache/llm_cache.sqlite`, so a second run of the grid makes no new LLM calls and differs only through the clustering initialization. Because the LLM is a quantized local model, keyphrases and descriptions may differ slightly across machines; the metric values in `results/` are those reported in the paper.

## Streamlit app

```bash
.venv/bin/streamlit run app.py
```

Pages: 1 data preparation, 2 single-run exploration, 3 batch experiments, 4 results and statistics.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q     # unit tests, LLM stubbed
.venv/bin/python scripts/smoke_test.py    # end-to-end smoke test, needs Ollama
```

## License

MIT License. Copyright (c) 2026 Kyunghoon Cho (Korea National Open University). See `LICENSE`.

A Korean description of the platform is in `README.ko.md`.

`data/samples/sample_bigkinds.csv` is a synthetic file (40 made-up headlines with fictitious press names) that only shows the expected column layout; it contains no BigKinds data.
