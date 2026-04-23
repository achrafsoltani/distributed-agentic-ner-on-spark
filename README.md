# Distributed NER on Spark — Reproducibility Package

Companion code and gold-set data for the paper *Distributed NER on Spark: A Teacher-Student Pipeline for Large-Scale Entity Extraction from Job Postings* (Soltani and Hanine, 2026).

This repository is **not** the paper. It is the minimum needed to reproduce the paper's numerical claims — including the eight Table 5 confidence intervals, the non-distillation B0 baseline, the §4.3.1 error-mode catalogue, the §4.3.2 sliding-window-inference ablation, the §4.3.3 chunked-window-training follow-up, and the §4.5 long-tail latency figure — against the released models, the annotated gold set, and the on-request artefacts documented at the end.

A reproducibility walk on a fresh clone (2026-04-22) verified that all four main scripts PASS or PASS-within-bootstrap-noise; see HEAD `c201656` and the patches that landed in `2bf3455` + `59e2287` + `c201656`.

## Published models

The six student models trained and evaluated in the paper are on HuggingFace Hub:

| Student | Architecture | Teacher | Hub identifier |
|---|---|---|---|
| S1 | spaCy `en_core_web_lg` | Claude Sonnet 4.6 | [`AchrafSoltani/spacy-lg-jobposting-ner-sonnet-v1`](https://huggingface.co/AchrafSoltani/spacy-lg-jobposting-ner-sonnet-v1) |
| **S2** | spaCy `en_core_web_lg` | Claude Haiku 4.5 | [`AchrafSoltani/spacy-lg-jobposting-ner-haiku-v1`](https://huggingface.co/AchrafSoltani/spacy-lg-jobposting-ner-haiku-v1) **(production pick)** |
| S3 | `jjzha/jobbert-base-cased` | Claude Sonnet 4.6 | [`AchrafSoltani/jobbert-ner-sonnet-v1`](https://huggingface.co/AchrafSoltani/jobbert-ner-sonnet-v1) |
| S4 | `jjzha/jobbert-base-cased` | Claude Haiku 4.5 | [`AchrafSoltani/jobbert-ner-haiku-v1`](https://huggingface.co/AchrafSoltani/jobbert-ner-haiku-v1) |
| S5 | JobBERT → ONNX int8 | Claude Sonnet 4.6 | [`AchrafSoltani/jobbert-ner-sonnet-v1-onnx`](https://huggingface.co/AchrafSoltani/jobbert-ner-sonnet-v1-onnx) |
| S6 | JobBERT → ONNX int8 | Claude Haiku 4.5 | [`AchrafSoltani/jobbert-ner-haiku-v1-onnx`](https://huggingface.co/AchrafSoltani/jobbert-ner-haiku-v1-onnx) |

## What's in this repository

```
pipeline/
├── annotation/
│   └── ANNOTATION-GUIDELINES.md              — rules the 516-posting gold set follows (8 entity types, R1–R8, edge cases)
├── labels-public/
│   ├── sonnet/gold-predictions.jsonl         — 516 Sonnet teacher predictions on the gold set (sufficient to reproduce the teacher cells of Table 5)
│   └── haiku/gold-predictions.jsonl          — 515 Haiku teacher predictions on the gold set
├── output-sample/
│   └── scaler-sample-10k.parquet             — 10,000-posting reservoir sample of the 1.3M Phase 8 output (sample-mode default for longtail_scatter.py; sample stats match population to within sampling noise)
├── scripts/
│   ├── bootstrap_ci.py                       — reproduces the 95% bootstrap CIs (Tables 4–5); Hub-aware: pulls student weights on demand if not present locally
│   ├── evaluate_student.py                   — reusable per-student evaluation (single-shot inference)
│   ├── evaluate_student_sliding.py           — sliding-window inference predictor (§4.3.2)
│   ├── bootstrap_ci_sliding.py               — 4-cell bootstrap CIs for sliding-window inference on S3/S4/S5/S6 (§4.3.2)
│   ├── train_jobbert_chunked.py              — chunked-window training (§4.3.3); self-contained, ~470 lines
│   ├── bootstrap_ci_chunked.py               — bootstrap CIs for chunked-trained S3'/S4' weights (§4.3.3); self-contained
│   ├── gold_only_baseline.py                 — reproduces the B0 non-distillation baseline (Table 5 top row)
│   ├── s2_error_analysis.py                  — reproduces the S2 error-mode catalogue (§4.3.1); Hub-aware
│   └── longtail_scatter.py                   — reproduces the long-tail latency figure (§4.5); uses output-sample/ by default, falls back to full Phase 8 outputs if available locally
└── training/
    ├── haiku/gold.parquet                    — 515 gold postings on the Haiku dev intersection
    ├── sonnet/gold.parquet                   — 516 gold postings on the Sonnet dev intersection
    └── experiments/specs/*.yaml              — training hyperparameter specs for the six v1 students plus the chunked S3'/S4' v2 specs

requirements.txt                              — pinned Python dependencies (includes spacy-lookups-data + huggingface_hub for the Hub-aware loaders)
.env.example                                  — environment-variable template (HuggingFace token)
```

## Reproducing the paper's numbers

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU torch first
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# Eight Table 5 student/teacher CIs (Hub-aware: downloads ~2 GB of model weights on first run, cached thereafter)
python -m pipeline.scripts.bootstrap_ci

# B0 non-distillation baseline (5-fold CV + bootstrap), Table 5 top row
python -m pipeline.scripts.gold_only_baseline

# S2 error-mode catalogue (§4.3.1, Table 6)
python -m pipeline.scripts.s2_error_analysis

# Long-tail latency scatter (§4.5, Fig 5) — uses the shipped 10K sample by default
python -m pipeline.scripts.longtail_scatter
```

Sliding-window inference ablation (§4.3.2) — runs end-to-end from this package and the Hub-published v1 weights:

```bash
python -m pipeline.scripts.bootstrap_ci_sliding --source hub
```

Chunked-window training (§4.3.3) — script ships in this package; runnable end-to-end against the chunked checkpoints (which are an on-request artefact, see below). Documents the methodology for any reviewer who wants to re-train from teacher labels.

```bash
python -m pipeline.scripts.train_jobbert_chunked --spec pipeline/training/experiments/specs/s4_jobbert_chunked_haiku.yaml
python -m pipeline.scripts.bootstrap_ci_chunked --source local
```

## On-request artefacts

These are not in the repo; contact the authors:

- **Chunked-trained S3$'$ and S4$'$ weights** (412 MB each) — can be regenerated from `train_jobbert_chunked.py` + the curated teacher train/dev parquets (also on request)
- **Curated teacher train/dev parquets** — `pipeline/training/{sonnet,haiku}/{train,dev}.parquet`, the 80/10/10 split outputs of the Phase 5b curator
- **Phase 8 cluster-lifecycle orchestrator** + cloud-init scripts
- **Full raw Phase 8 scaler output parquet** (~647 MB across 43 files) — the `output-sample/scaler-sample-10k.parquet` is a reservoir sample of this
- **Analyst intermediate outputs**

The 1.3M-posting primary corpus is publicly available at [Kaggle — asaniczka/1-3m-linkedin-jobs-and-skills-2024](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024).

## Licensing

- **Code** (`pipeline/scripts/`): Apache 2.0 — see [`LICENSE-CODE`](LICENSE-CODE).
- **Research content** (annotation guidelines, gold set, gold-set teacher predictions): CC BY-NC 4.0 — see [`LICENSE-RESEARCH`](LICENSE-RESEARCH).
- **Model weights on HuggingFace Hub**: CC BY-NC 4.0 — research and non-commercial evaluation.

## Citation

```bibtex
@unpublished{soltani2026distilledner,
  author = {Achraf Soltani and Mohamed Hanine},
  title  = {Distributed NER on Spark: A Teacher-Student Pipeline for Large-Scale Entity Extraction from Job Postings},
  year   = {2026},
  note   = {Preprint, April 2026},
}
```

Paper venue will be linked here once published.

## Contact

- Achraf Soltani — [www.achrafsoltani.com](https://www.achrafsoltani.com)
- Prof. Mohamed Hanine — Université Chouaib Doukkali, El Jadida, Morocco
