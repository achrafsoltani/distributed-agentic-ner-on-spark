# Distributed NER on Spark — Reproducibility Package

Companion code and gold-set data for the paper *Distributed NER on Spark: A Teacher-Student Pipeline for Large-Scale Entity Extraction from Job Postings* (Soltani and Hanine, 2026).

This repository is **not** the paper. It is the minimum needed to reproduce the paper's numerical claims against the released models and annotated gold set.

## Published models

The six student models trained and evaluated in the paper are on HuggingFace Hub:

| Student | Architecture | Teacher | Hub identifier |
|---|---|---|---|
| S1 | spaCy `en_core_web_lg` | Claude Sonnet 4.6 | [`AchrafSoltani/spacy-lg-jobposting-ner-sonnet-v1`](https://huggingface.co/AchrafSoltani/spacy-lg-jobposting-ner-sonnet-v1) |
| S2 | spaCy `en_core_web_lg` | Claude Haiku 4.5 | [`AchrafSoltani/spacy-lg-jobposting-ner-haiku-v1`](https://huggingface.co/AchrafSoltani/spacy-lg-jobposting-ner-haiku-v1) (production pick) |
| S3 | `jjzha/jobbert-base-cased` | Claude Sonnet 4.6 | [`AchrafSoltani/jobbert-ner-sonnet-v1`](https://huggingface.co/AchrafSoltani/jobbert-ner-sonnet-v1) |
| S4 | `jjzha/jobbert-base-cased` | Claude Haiku 4.5 | [`AchrafSoltani/jobbert-ner-haiku-v1`](https://huggingface.co/AchrafSoltani/jobbert-ner-haiku-v1) |
| S5 | JobBERT → ONNX int8 | Claude Sonnet 4.6 | [`AchrafSoltani/jobbert-ner-sonnet-v1-onnx`](https://huggingface.co/AchrafSoltani/jobbert-ner-sonnet-v1-onnx) |
| S6 | JobBERT → ONNX int8 | Claude Haiku 4.5 | [`AchrafSoltani/jobbert-ner-haiku-v1-onnx`](https://huggingface.co/AchrafSoltani/jobbert-ner-haiku-v1-onnx) |

## What's in this repository

```
pipeline/
├── annotation/
│   └── ANNOTATION-GUIDELINES.md      — rules the 516-posting gold set follows
├── scripts/
│   ├── bootstrap_ci.py               — reproduces the 95% bootstrap CIs (paper §3.8, Tables 4–5)
│   ├── evaluate_student.py           — reproduces per-type F1/P/R numbers against gold
│   ├── gold_only_baseline.py         — reproduces the B0 non-distillation baseline (Table 5)
│   ├── longtail_scatter.py           — reproduces the per-posting latency scatter (Fig 7)
│   └── s2_error_analysis.py          — reproduces the S2 error-mode catalogue (§4.3.1)
└── training/
    ├── haiku/gold.parquet            — 515 gold postings on the Haiku dev intersection
    ├── sonnet/gold.parquet           — 516 gold postings on the Sonnet dev intersection
    └── experiments/specs/*.yaml      — training hyperparameter specs for the six students

requirements.txt                      — pinned Python dependencies
.env.example                          — environment-variable template (HuggingFace token)
```

## Reproducing the paper's numbers

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# 95% bootstrap CIs for all eight F1 cells in Tables 4 and 5
python -m pipeline.scripts.bootstrap_ci

# B0 non-distillation baseline (5-fold CV + bootstrap), Table 5 top row
python -m pipeline.scripts.gold_only_baseline

# S2 error-mode catalogue (§4.3.1, Table 6)
python -m pipeline.scripts.s2_error_analysis
```

Two scripts additionally require the full Phase 8 output parquet (338 MB) and the [Kaggle 1.3M LinkedIn Jobs corpus](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024):

```bash
# Long-tail latency scatter (Fig 7). Needs the Phase 8 raw parquet.
python -m pipeline.scripts.longtail_scatter
```

## Licensing

- **Code** (`pipeline/scripts/`): Apache 2.0 — see [`LICENSE-CODE`](LICENSE-CODE).
- **Research content** (annotation guidelines, gold set): CC BY-NC 4.0 — see [`LICENSE-RESEARCH`](LICENSE-RESEARCH).
- **Model weights on HuggingFace Hub**: CC BY-NC 4.0 — research and non-commercial evaluation.

## Citation

```bibtex
@unpublished{soltani2026distilledner,
  author = {Achraf Soltani and Mohamed Hanine},
  title  = {Distributed NER on Spark: A Teacher-Student Pipeline for Large-Scale Entity Extraction from Job Postings},
  year   = {2026},
  note   = {Advisor: Prof.\ Hanine Mohamed},
}
```

Paper venue will be linked here once published.

## Contact

Achraf Soltani — [www.achrafsoltani.com](https://www.achrafsoltani.com)
Advisor: Prof. Hanine Mohamed
