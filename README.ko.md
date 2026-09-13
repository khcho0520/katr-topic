# TopicNew — 토픽모델링 개선 모델 검증 플랫폼

QualIT(Kapoor et al., [arXiv:2409.15626](https://arxiv.org/abs/2409.15626))에서 아이디어를 얻은
개선 토픽모델을 제안하고, 레거시 모델(BERTopic, LDA, NMF) 대비 우수성을 **두 데이터 체제**
(언론 헤드라인 = 다수 짧은 텍스트 / 논문 = 소수 긴 텍스트)에서 논문 게재 수준으로 검증하는
Streamlit 실험 플랫폼.

## 구성

- **로컬 Ollama**: 임베딩 `bge-m3:latest`, LLM `qwen3.5:9b`(키프레이즈/라벨), `qwen3.5:27b`(judge)
- **레거시 모델**: BERTopic(UMAP+HDBSCAN), LDA, NMF — 공유 bge-m3 임베딩·공유 Kiwi 토큰 사용
- **QualIT 계열**: 논문 재현 + ablation 변형 (환각필터/클러스터링/표현/키워드/병합 교체 가능)

## 실행

```bash
cd /Volumes/project/TopicNew
.venv/bin/streamlit run app.py
```

### 페이지

1. **데이터 준비** — BigKinds Excel/CSV 또는 논문 PDF/DOCX 업로드 → Kiwi 토큰화 + bge-m3 임베딩 사전계산 + 저장.
   **MCPanal 월별 세션 탭**: 경기도 언론분석의 월별 '대한민국 전체' 임베딩 세션(ChromaDB)을
   복수 선택·병합해 가져옴 — 저장된 bge-m3 임베딩을 그대로 재사용해 재임베딩 없음
2. **단일 실행 탐색** — 모델 1개 × 시드 1개 실행, 토픽·지표·UMAP 지도 질적 검토.
   결과를 **세션으로 저장/복원** 가능 (설정 스냅샷+토픽 결과+지표, `runs/_workspaces/`)
3. **배치 실험** — (설정 × 시드) 그리드 실행. 중단해도 같은 실험 ID로 재개. LLM 출력은 캐시로 시드 간 재사용
4. **결과·통계·내보내기** — 평균±표준편차 표, 쌍체 t-검정/Wilcoxon/Cohen's d/Holm 보정, CSV/Markdown/LaTeX 내보내기

### 헤드리스 실행 (대규모 그리드)

```bash
.venv/bin/python scripts/run_experiment.py --list-corpora
.venv/bin/python scripts/run_experiment.py \
    --corpus-hash <해시> \
    --models bertopic,lda,nmf,qualit_paper,qualit_improved \
    --seeds 42,43,44,45,46 --judge
```

## 공정 평가 프로토콜 (논문 방법론)

- **공유 임베딩**: 모든 임베딩 기반 모델·지표가 동일한 bge-m3 행렬 사용
- **공유 토크나이저**: NPMI/Diversity는 모든 모델의 키워드를 Kiwi 명사 토큰으로 정규화 후 계산
- **시드 의미론**: LLM 단계(키프레이즈 추출 temp 0.2, 라벨링)는 시드 무관 → 캐시 재사용.
  클러스터링(KMeans/UMAP 초기화)만 시드별 변화. → "시드 변동은 확률적 클러스터링을 포괄하며,
  LLM 추출은 고정 후 재사용"으로 기술
- **K 공정성**: `n_topics_found`를 지표로 기록. **공통 조건 기능**(페이지 3 '⚖️ 공통 조건' /
  CLI `--n-topics`, `--bertopic-mcs`, `--qualit-max-llm-docs`)으로 모든 모델에 동일 K·
  BERTopic min_cluster_size·QualIT LLM 상한을 일괄 적용 — 적용된 설정은 결과 표에
  접미사(K=…, mcs=…)로 구분됨. BERTopic은 nr_topics 축소 방식이라 K가 근사치일 수 있음.
  **자동 제안**(페이지 3 '🔮 데이터 기반 자동 제안' / CLI `--auto-conditions`,
  core/experiments/recommend.py): 임베딩 표본(≤3,000건)의 코사인 실루엣 k 스캔 +
  관용 규칙(최고 실루엣의 90% 이상 중 최대 k — 굵은 분할 붕괴 방지)으로 공통 K를 추정하고,
  mcs=n/(K×20), QualIT LLM 상한=K×50(200~1,000)을 함께 제안. 전 과정 결정적(고정 시드),
  근거와 k별 실루엣 곡선을 함께 표시
- **분석 단위 통일**: 논문 체제는 청크 단위로 모델링(두 체제를 구조적으로 동일화), 논문별 집계는 보고용
- **대량 코퍼스 보호**: QualIT 계열은 `max_llm_docs`(기본 500) 초과 시 임베딩 층화 샘플링으로
  대표 문서만 LLM 키프레이즈 추출(고정 시드 → 시드 간 캐시 재사용), 나머지 문서는 최근접 토픽 할당.
  Ollama 연속 호출 실패 시 서킷브레이커가 즉시 중단(성공분은 캐시 보존)

## 지표

| 분류 | 지표 |
|---|---|
| 임베딩 | Coherence(센트로이드 코사인), Separation, Silhouette(할당/노이즈포함) |
| 어휘 | NPMI, Topic Diversity (Kiwi 정규화) |
| 외적 타당도 | NMI, ARI, Purity — 기사 카테고리 대분류(BigKinds `통합 분류1`/MCPanal 메타)를 정답 레이블로 자동 계산 (레이블 보유 코퍼스만) |
| 재현 안정성 | 시드 쌍별 할당 ARI (같은 설정, 페이지 4 별도 표) — 초기화 민감도 정량화 |
| 구조 | Coverage(할당률), #Topics, 토픽 크기 통계, Runtime |
| 질적(선택) | LLM-as-judge 4차원: 명확성·일관성·구분성·근거성 (캐시됨) |

각 지표의 정의·타당성 근거·선행연구·한계는 **[docs/METRICS.md](docs/METRICS.md)** 참조
(논문 평가 방법 절 작성용 인용 문헌 포함).

## 프리셋 (core/models/configs.py)

**`katr_core`** — **제안 모델 KATR 최종형**(Keyphrase-Anchored Topic Refinement, core/models/katr_model.py):
UMAP+문서 클러스터링(BERTopic 구조) + 센트로이드 재할당 + 코어 문서 c-TF-IDF 키워드 + LLM 라벨,
전수 커버리지 내장. 실측(3개월 12만 건, K=20): BERTopic-full과 coherence·purity 동률,
재현 안정성 ARI +0.16 우세 / `katr`(α=0.6 앵커) `katr_noanchor`(α=1 대조군) — ablation 짝 /
`bertopic` `bertopic_full`(아웃라이어 전수할당 — 커버리지 교란 제거) `lda` `nmf` — 베이스라인 /
`qualit_paper`(논문 재현) `qualit_joint`(QualIT 계열 실측 최고) — 참조 모델.
그 외 ablation 변형(hdbscan/ctfidf/percentile/llm_verify/refine 등)은 페이지 2 폼에서 수동 구성

## 검증

```bash
.venv/bin/python -m pytest tests/ -q       # 단위 테스트 (Ollama 불필요, LLM 스텁)
.venv/bin/python scripts/smoke_test.py      # 전체 파이프라인 스모크 (Ollama 필요)
```

스모크를 2회 연속 실행하면 두 번째는 "신규 LLM HTTP 호출 0건"이어야 한다 (캐시 검증).

## 디렉토리

```
core/        Streamlit 무관 순수 라이브러리 (data/llm/models/metrics/experiments)
pages/       Streamlit 페이지 (얇은 뷰)
scripts/     헤드리스 CLI (run_experiment, smoke_test)
cache/       임베딩·LLM sqlite 캐시, 준비된 코퍼스 (gitignore)
runs/        실험 기록 JSON (gitignore)
```

## 환경

- Python 3.11 (`.venv`), 의존성은 `requirements.txt`
- `OLLAMA_HOSTS` 환경변수로 멀티호스트 폴백 (쉼표 구분, 첫 번째 우선 — 예: Tailscale MacBook → localhost)
- 참고 구현: `/Volumes/project/MCPanal` '토픽모델링 검증' 기능 (로직 이식, 의존성 없음)
