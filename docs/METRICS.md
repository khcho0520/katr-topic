# 평가지표 타당성 문서 (Metric Validity & Rationale)

TopicNew가 계산하는 모든 모델 간 비교 지표의 **정의, 구현, 타당성 근거(선행연구), 해석 방법, 한계**를 정리한다.
논문의 평가 방법(Evaluation) 절 작성 시 이 문서의 근거 문헌을 인용할 수 있다.

> 구현 위치: `core/metrics/` (embedding_metrics, lexical_metrics, external, coverage, llm_judge, suite)
> 및 `core/experiments/stats.py` (안정성, 통계 검정)

---

## 0. 공정 평가 프로토콜 (전제)

모든 지표는 다음 통제 아래 계산된다. 지표 자체보다 이 통제가 비교 타당성의 전제다.

| 통제 | 내용 | 근거 |
|---|---|---|
| 공유 임베딩 | 모든 모델·임베딩 지표가 동일한 bge-m3 행렬 사용 | 표현 공간이 다르면 임베딩 기반 지표는 비교 불가. 표준화 평가 프레임워크 OCTIS도 동일 사상 (Terragni et al., 2021, EACL demo: *OCTIS: Comparing and Optimizing Topic Models is Simple!*) |
| 공유 토크나이저 | NPMI/Diversity는 모든 모델의 키워드를 Kiwi 명사 토큰으로 정규화 후 계산 | 표면형(어절 vs 형태소) 불일치가 어휘 지표를 왜곡하는 것을 차단 |
| 공통 K 옵션 | 토픽 수를 통일해 K 교란 제거 (`apply_common_conditions`) | Diversity·Separation·Purity는 K에 민감 (아래 각 절 참조) |
| 시드 의미론 | LLM 단계는 temp 0.2 고정·캐시 재사용, 클러스터링만 시드 의존 | 반복실험의 분산 출처를 "확률적 클러스터링 초기화"로 특정 |

---

## 1. 어휘 일관성 — NPMI (Normalized Pointwise Mutual Information)

**구현** (`lexical_metrics.py::npmi_for_topic`): 토픽 상위 키워드 쌍 (w₁,w₂)에 대해
NPMI(w₁,w₂) = PMI(w₁,w₂) / (−log p(w₁,w₂)), 확률은 **코퍼스 내부의 문서 단위 공기(co-occurrence)** 문서빈도로 추정.
토픽 점수는 유효 쌍의 평균, 모델 점수는 토픽 평균. 범위 [−1, 1].

**타당성 근거**
- NPMI 정식화: **Bouma (2009)**, *Normalized (Pointwise) Mutual Information in Collocation Extraction*, GSCL.
- 토픽 일관성 자동 평가로서의 검증: **Lau, Newman & Baldwin (2014)**, *Machine Reading Tea Leaves: Automatically Evaluating Topic Coherence and Topic Model Quality*, EACL — NPMI가 사람의 토픽 품질 판단과 가장 높은 상관을 보이는 자동 지표군에 속함을 보고. 토픽모델링 논문의 사실상 표준 지표.
- 자동 일관성 지표의 원류: **Newman et al. (2010)**, *Automatic Evaluation of Topic Coherence*, NAACL.
- 일관성 지표 공간의 체계화: **Röder, Both & Hinneburg (2015)**, *Exploring the Space of Topic Coherence Measures*, WSDM — C_v, C_npmi 등 조합 프레임워크.

**본 구현의 선택과 방어 논리**
- 슬라이딩 윈도우(Lau et al. 방식)가 아닌 **문서 단위 공기**를 쓴다. 이는 **Mimno et al. (2011)**, *Optimizing Semantic Coherence in Topic Models*, EMNLP (UMass coherence)의 문서 공기 전통을 따른 것으로, **헤드라인처럼 문서가 짧은 체제에서는 문서≈윈도우가 되어 두 방식이 수렴**한다. 논문에는 "document-level co-occurrence 기반 NPMI"로 명시할 것.
- 외부 참조 코퍼스(Wikipedia 등) 대신 분석 코퍼스 내부 통계를 쓴다 — 한국어 뉴스/논문 도메인에서 외부 코퍼스 통계는 도메인 불일치를 일으킴 (Lau et al. 2014도 참조 코퍼스의 도메인 정합을 강조).

**해석·한계**: 높을수록 키워드들이 실제로 같이 등장하는 "말이 되는" 토픽. 키워드가 코퍼스에 드물면 유효 쌍 부족으로 0이 나올 수 있음(min_df=2). 긴 문서(논문 청크)에서는 문서 공기가 관대해져 값이 후해지는 경향 — 체제 간 절대값 비교는 하지 말 것.

---

## 2. 어휘 다양성 — Topic Diversity

**구현** (`lexical_metrics.py::topic_diversity`): 전 토픽 상위 top-k(기본 10) 키워드 중 고유 키워드 비율. 범위 [0, 1].

**타당성 근거**
- **Dieng, Ruiz & Blei (2020)**, *Topic Modeling in Embedding Spaces*, TACL (ETM 논문) — "topic diversity = 전 토픽 상위 25단어 중 고유 단어 비율"로 정식 도입. 이후 신경 토픽모델 논문들의 표준 보조 지표.
- QualIT 원논문 (**Kapoor et al., 2024**, arXiv:2409.15626)도 coherence와 diversity를 주 지표로 사용 — 본 플랫폼이 같은 축을 유지해 직접 비교 가능.

**해석·한계**: 낮으면 토픽들이 같은 단어를 반복(중복 토픽 신호). **K가 작을수록 1.0이 쉽게 나옴** — 지난 실험에서 BERTopic(K=3)의 diversity 1.0이 그 예. 반드시 공통 K 조건에서 비교할 것. 일관성과 트레이드오프 관계이므로 NPMI와 쌍으로 보고해야 함 (Dieng et al.도 coherence×diversity 곱을 종합 지표로 제안).

---

## 3. 의미 일관성 — Embedding Coherence (센트로이드 코사인)

**구현** (`embedding_metrics.py::topic_coherence`): 토픽 멤버 문서 임베딩의 중심(centroid)과 각 멤버 간 코사인 유사도 평균. 모델 점수는 토픽 평균.

**타당성 근거**
- 클러스터 응집도(cohesion)의 표준 정식화 — 군집 타당성 지표 문헌의 intra-cluster similarity에 해당 (예: **Calinski & Harabasz, 1974**, *A dendrite method for cluster analysis*의 군집 내 분산 개념과 동일 계열).
- 임베딩 공간에서의 토픽 품질 평가는 임베딩 기반 토픽모델의 자연스러운 확장: **Grootendorst (2022)**, *BERTopic: Neural topic modeling with a class-based TF-IDF procedure*, arXiv:2203.05794. QualIT 원논문(Kapoor et al., 2024)도 임베딩 공간 coherence를 사용.
- 어휘 지표와의 상보성: NPMI는 표면형 공기를, 임베딩 coherence는 의미 유사성을 포착 — 단어가 달라도 의미가 같은 문서 묶음(짧은 한국어 헤드라인에서 흔함)을 평가 가능.

**해석·한계**: 높을수록 토픽 내 문서가 의미적으로 응집. **coverage와 반드시 함께 볼 것** — 아웃라이어를 버리는 모델(BERTopic)은 할당된 "쉬운" 문서만으로 계산되어 유리해짐 (지난 실험: BERTopic 0.585 @ coverage 53% vs QualIT 0.585 @ 100%). bge-m3 임베딩의 코사인 값 자체가 좁은 범위에 몰리므로 절대값보다 모델 간 차이로 해석.

---

## 4. 구분성 — Separation & Silhouette

**구현** (`embedding_metrics.py`):
- Separation: 토픽 센트로이드 쌍별 평균 코사인 거리(1−sim).
- Silhouette: **Rousseeuw (1987)**, *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis*, J. Comput. Appl. Math. — s(i) = (b−a)/max(a,b). 두 변형: 할당 문서만(`silhouette`) / 아웃라이어를 노이즈 토픽으로 포함(`silhouette_noise`).

**타당성 근거**
- Silhouette는 군집 타당성 내부 지표의 고전이자 표준 (Rousseeuw 1987, 인용 수만 회). 응집도(a)와 분리도(b)를 한 값에 결합해 coherence·separation의 종합 검증 역할.
- Separation(센트로이드 간 거리)은 **Davies & Bouldin (1979)**, *A Cluster Separation Measure*, IEEE TPAMI 계열의 군집 간 분리 개념을 코사인 공간에 단순화한 것.

**해석·한계**: 고차원 원 임베딩(1024차원)에서 silhouette 절대값은 0.0x대로 작게 나오는 것이 정상(차원의 저주로 거리 대비가 축소) — 모델 간 상대 비교로만 사용. Separation은 **K가 작을수록 유리** → 공통 K 필수. 두 변형을 같이 보고하면 "버린 문서 효과"를 분리할 수 있음: 할당-only는 BERTopic에 관대, 노이즈 포함은 커버리지 페널티 반영.

---

## 5. 외적 타당도 — NMI · ARI · Purity

**구현** (`external.py`): 기사 카테고리 대분류(BigKinds `통합 분류1` / MCPanal 메타의 `사회>사건_사고 | …` → `사회`)를 정답 분할로 삼아, 할당·레이블 모두 있는 문서에 대해 계산.

**타당성 근거**
- **NMI**: **Strehl & Ghosh (2002)**, *Cluster Ensembles — A Knowledge Reuse Framework for Combining Multiple Partitions*, JMLR — 분할 간 상호정보량 정규화. 군집-정답 비교의 표준.
- **ARI**: **Hubert & Arabie (1985)**, *Comparing Partitions*, Journal of Classification (Rand 1971의 우연 보정판) — 우연 일치를 기대값 0으로 보정, 군집 수 차이에 강건.
- **Purity**: **Manning, Raghavan & Schütze (2008)**, *Introduction to Information Retrieval*, Ch.16 — 토픽별 최빈 클래스 비율. 직관적 해석("이 토픽의 몇 %가 같은 카테고리인가") 제공.
- 지표 선택 주의: **Vinh, Epps & Bailey (2010)**, *Information Theoretic Measures for Clusterings Comparison: Variants, Properties, Normalization and Correction for Chance*, JMLR — NMI는 군집 수가 많을수록 우연히 커지는 편향이 있어 **우연 보정된 ARI를 병행 보고**해야 함. 본 플랫폼이 세 지표를 모두 내는 이유.
- 토픽모델을 문서 군집으로 보고 외적 레이블과 대조하는 평가는 단문 토픽모델링 문헌의 관행 (예: 20 Newsgroups purity/NMI 평가 전통; Yin & Wang 2014, *A Dirichlet Multinomial Mixture Model-based Approach for Short Text Clustering*, KDD).

**해석·한계**: "토픽 구조가 언론사 분류 체계라는 **외부 기준**과 얼마나 정합한가". 단, 카테고리는 하나의 참조 분할일 뿐 유일한 정답이 아님 — 카테고리와 다른 유의미한 토픽 구조(예: 사건 단위)가 낮은 NMI로 나올 수 있음. 따라서 외적 지표는 내적 지표와 **동률일 때의 타이브레이커/보조 근거**로 사용하고, 논문에는 "참조 분할(reference partition) 대비"로 표현할 것. Purity는 K가 클수록 자동 상승(Manning et al.) → 공통 K 필수. 다중 카테고리 기사(` | ` 구분)는 첫 항목 대분류만 사용함을 방법론에 명시.

---

## 6. 재현 안정성 — 시드 쌍별 ARI (Stability)

**구현** (`stats.py::stability_table`): 같은 설정의 시드 쌍 (sᵢ, sⱼ)에 대해 두 실행의 문서 할당 ARI를 계산(두 실행 모두 할당된 문서만), 쌍 평균·표준편차·최솟값 보고.

**타당성 근거**
- 군집 안정성을 모델 선택·검증 기준으로 쓰는 이론적 정립: **Lange, Roth, Braun & Buhmann (2004)**, *Stability-Based Validation of Clustering Solutions*, Neural Computation; 개관: **von Luxburg (2010)**, *Clustering Stability: An Overview*, Foundations and Trends in Machine Learning.
- 토픽모델에의 적용: **Greene, O'Callaghan & Cunningham (2014)**, *How Many Topics? Stability Analysis for Topic Models*, ECML-PKDD — 실행 간 일치도로 토픽모델의 신뢰성·적정 K를 평가. LDA의 실행 간 불안정성은 널리 보고된 문제.
- 일치도 척도로 ARI를 쓰는 이유: 우연 보정 + **토픽 ID 순열 불변**(같은 분할이면 ID가 달라도 1.0) — 실행 간 비교에 필수 성질 (Hubert & Arabie 1985).

**해석·한계**: 1.0 = 시드와 무관하게 동일 구조(예: 실측 NMF 0.997), 0 근처 = 초기화마다 다른 결과(예: 실측 LDA 0.028). **"이 방법의 결과를 신뢰하고 재현할 수 있는가"를 직접 정량화** — 제안 모델이 QualIT 재현의 토픽 수 불안정(±3.03)을 개선했음을 보이는 핵심 지표. 한계: 안정적이라고 좋은 토픽은 아님(퇴화 해도 안정적일 수 있음) — 품질 지표와 반드시 병행.

---

## 7. 구조 지표 — Coverage · #Topics · 크기 분포 · Runtime

**구현** (`coverage.py`): 할당률(assignments ≥ 0 비율), 발견 토픽 수, 토픽 크기 std/min/max, 실행 시간.

**타당성 근거**
- HDBSCAN 계열(BERTopic)은 노이즈 포인트를 명시적으로 버림: **McInnes, Healy & Astels (2017)**, *hdbscan: Hierarchical density based clustering*, JOSS; **Grootendorst (2022)** — 따라서 coverage는 임베딩·외적 지표의 **해석 조건**으로 반드시 병기해야 하는 지표 (3절 한계 참조).
- `n_topics_found`는 K-free 모델(HDBSCAN)과 고정 K 모델의 비교 조건을 기록하는 메타 지표 — 공정성 프로토콜의 감사 장치.

**해석**: Runtime은 cold(첫 실행, LLM 실호출)와 warm(캐시 재사용)을 구분해 보고할 것 — LLM 파이프라인의 캐시 구조상 시드 평균 runtime은 오해를 유발함.

---

## 8. 질적 평가 — LLM-as-Judge (선택)

**구현** (`llm_judge.py`): 토픽 라벨·키워드·대표 문서를 제시하고 4차원(명확성·일관성·구분성·근거성) 1~5점 루브릭으로 로컬 LLM(qwen3.5:27b)이 채점. (토픽 집합, 모델, 루브릭 버전) 해시로 캐시.

**타당성 근거**
- 사람의 토픽 해석 가능성 평가의 원류: **Chang, Boyd-Graber et al. (2009)**, *Reading Tea Leaves: How Humans Interpret Topic Models*, NeurIPS — 자동 지표가 사람 판단과 어긋날 수 있음을 보이고 사람 평가(word/topic intrusion)의 필요성을 정립. LLM judge는 이 사람 평가의 확장 가능한 대체재.
- LLM 판정자의 타당성: **Zheng et al. (2023)**, *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, NeurIPS — LLM 판정이 사람 선호와 높은 일치(>80%)를 보임.
- 토픽모델 평가에의 직접 적용: **Stammbach et al. (2023)**, *Revisiting Automated Topic Model Evaluation with Large Language Models*, EMNLP — LLM이 토픽 일관성 평가·적정 토픽 수 판단에서 사람 판단과 정합함을 보고.
- 루브릭 4차원은 MCPanal의 검증된 평가축을 계승 (명확성=라벨 이해도, 일관성=대표문서 내부 통일성, 구분성=토픽 간 경계, 근거성=라벨-문서 정합).

**해석·한계**: LLM judge는 자기 선호 편향·프롬프트 민감성이 알려져 있음(Zheng et al.의 position bias 등) — 판정 모델(qwen3.5:27b)을 파이프라인 모델(9b)과 분리했고, 온도 0.1 고정. 그래도 주 지표가 아닌 **보조 질적 근거**로 사용하고, 논문에서는 소표본 사람 평가로 스팟체크하는 것을 권장.

---

## 9. 통계 검정 — 쌍체 t / Wilcoxon / Cohen's d / Holm

**구현** (`stats.py::paired_tests`): 동일 시드를 쌍으로 paired t-test와 Wilcoxon signed-rank를 병행, Cohen's d(쌍차 기반) 효과크기, 다중비교는 Holm 보정.

**타당성 근거**
- 알고리즘 비교에서 비모수 검정 권장: **Demšar (2006)**, *Statistical Comparisons of Classifiers over Multiple Data Sets*, JMLR — 정규성 가정이 약한 소표본 비교에서 Wilcoxon signed-rank를 권장. t-검정과 병행 보고로 강건성 확인.
- **Holm (1979)**, *A Simple Sequentially Rejective Multiple Test Procedure*, Scandinavian Journal of Statistics — Bonferroni보다 검정력이 높은 균일 우세 보정.
- 효과크기 보고: **Cohen (1988)**, *Statistical Power Analysis for the Behavioral Sciences* — p값만으로는 효과의 크기를 알 수 없음. 관례 기준 0.2/0.5/0.8.

**해석·한계**: 쌍은 "같은 시드 = 같은 클러스터링 초기화 조건"으로 구성 — 시드 외 조건이 모두 통제되어야 유효(공정 프로토콜 전제). 시드 5개면 Wilcoxon의 최소 유의 가능 p≈0.0625이므로, 강한 주장에는 시드 8~10개를 권장.

---

## 10. 종합: 지표 체계의 3면 구조

| 면 | 지표 | 답하는 질문 |
|---|---|---|
| 내적 품질 | NPMI, Diversity, Emb-Coherence, Separation, Silhouette | 토픽이 그 자체로 좋은가 |
| 외적 타당도 | NMI, ARI, Purity (참조 분할 대비) | 외부 기준과 정합하는가 |
| 재현 안정성 | 시드 쌍별 ARI | 결과를 신뢰·재현할 수 있는가 |

+ 조건 지표(Coverage, #Topics)로 해석 조건을 병기하고, 쌍체 검정으로 차이의 통계적 유의성을 확인한다.
어느 한 면의 우위만으로 "더 나은 모델"을 주장하지 않는 것이 이 설계의 취지다 — 예컨대 지난 실험처럼
"QualIT: NPMI 우위 + 전수 커버리지, BERTopic: 임베딩 구분성 우위(단 53% 커버리지)"를 그대로 보고한다.

---

### 참고문헌 목록 (요약)

- Bouma, G. (2009). Normalized (Pointwise) Mutual Information in Collocation Extraction. *GSCL*.
- Calinski, T., & Harabasz, J. (1974). A dendrite method for cluster analysis. *Communications in Statistics*.
- Chang, J., Boyd-Graber, J., et al. (2009). Reading Tea Leaves: How Humans Interpret Topic Models. *NeurIPS*.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.).
- Davies, D. L., & Bouldin, D. W. (1979). A Cluster Separation Measure. *IEEE TPAMI*.
- Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. *JMLR*, 7.
- Dieng, A. B., Ruiz, F. J. R., & Blei, D. M. (2020). Topic Modeling in Embedding Spaces. *TACL*, 8.
- Greene, D., O'Callaghan, D., & Cunningham, P. (2014). How Many Topics? Stability Analysis for Topic Models. *ECML-PKDD*.
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv:2203.05794*.
- Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics*, 6(2).
- Hubert, L., & Arabie, P. (1985). Comparing Partitions. *Journal of Classification*, 2.
- Kapoor, S., Gil, A., Bhaduri, S., Mittal, A., & Mulkar, R. (2024). Qualitative Insights Tool (QualIT): LLM Enhanced Topic Modeling. *arXiv:2409.15626*. ✅ [서지 확인됨](https://arxiv.org/abs/2409.15626)
- Lange, T., Roth, V., Braun, M. L., & Buhmann, J. M. (2004). Stability-Based Validation of Clustering Solutions. *Neural Computation*, 16(6).
- Lau, J. H., Newman, D., & Baldwin, T. (2014). Machine Reading Tea Leaves: Automatically Evaluating Topic Coherence and Topic Model Quality. *EACL*.
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. (Ch. 16)
- McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical density based clustering. *JOSS*, 2(11).
- Mimno, D., et al. (2011). Optimizing Semantic Coherence in Topic Models. *EMNLP*.
- Newman, D., et al. (2010). Automatic Evaluation of Topic Coherence. *NAACL*.
- Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the Space of Topic Coherence Measures. *WSDM*.
- Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *J. Comput. Appl. Math.*, 20.
- Stammbach, D., Zouhar, V., Hoyle, A., Sachan, M., & Ash, E. (2023). Revisiting Automated Topic Model Evaluation with Large Language Models. *EMNLP 2023*, pp. 9348–9357. ✅ [서지 확인됨](https://aclanthology.org/2023.emnlp-main.581/)
- Strehl, A., & Ghosh, J. (2002). Cluster Ensembles — A Knowledge Reuse Framework for Combining Multiple Partitions. *JMLR*, 3.
- Terragni, S., et al. (2021). OCTIS: Comparing and Optimizing Topic Models is Simple! *EACL (demo)*.
- Vinh, N. X., Epps, J., & Bailey, J. (2010). Information Theoretic Measures for Clusterings Comparison. *JMLR*, 11.
- von Luxburg, U. (2010). Clustering Stability: An Overview. *Foundations and Trends in Machine Learning*, 2(3).
- Yin, J., & Wang, J. (2014). A Dirichlet Multinomial Mixture Model-based Approach for Short Text Clustering. *KDD*.
- Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS*.

> ⚠️ 인용 전 확인: 서지 정보는 작성 시점 지식 기반이므로, 논문 투고 전 각 문헌의 정확한
> 서지사항(연도·학회·페이지)을 Google Scholar 등에서 재확인할 것.
