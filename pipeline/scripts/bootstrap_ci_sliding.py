"""
Posting-level bootstrap CI for the four JobBERT/ONNX students, evaluated with
sliding-window inference rather than the single-shot 512-token pipeline call
used in pipeline.scripts.bootstrap_ci.

Companion to bootstrap_ci.py. Same SEED (42), same N_BOOTSTRAP (10,000),
same posting-level resampling --- only the predictor changes. Use the
two outputs together to populate the sliding-window addendum to Table 5.

Sources
-------
--source hub (default): pulls the public Hugging Face Hub repos under
    AchrafSoltani/*. No auth needed; avoids staging model weights on S3.
--source local:         uses Phase 6 model directories under
    pipeline/training/experiments/outputs/.

Usage
-----
    python -m pipeline.scripts.bootstrap_ci_sliding --source hub
    python -m pipeline.scripts.bootstrap_ci_sliding --source local

Writes pipeline/training/bootstrap_ci_sliding.json with one entry per cell
(s3_jobbert_sonnet, s4_jobbert_haiku, s5_jobbert_onnx_sonnet,
s6_jobbert_onnx_haiku) plus the sliding-window config block.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.scripts.bootstrap_ci import (
    _point_f1,
    bootstrap_f1,
    per_posting_contributions,
)
from pipeline.scripts.evaluate_student_sliding import build_sliding_predictor

ROOT = Path(__file__).resolve().parents[2]
GOLD_HAIKU = ROOT / "pipeline/training/haiku/gold.parquet"
GOLD_SONNET = ROOT / "pipeline/training/sonnet/gold.parquet"
OUT = ROOT / "pipeline/training/bootstrap_ci_sliding.json"

N_BOOTSTRAP = 10_000
SEED = 42
WINDOW_TOKENS = 450
STRIDE_TOKENS = 225

HUB_PATHS: dict[str, tuple[str, str]] = {
    "s3_jobbert_sonnet":      ("jobbert", "AchrafSoltani/jobbert-ner-sonnet-v1"),
    "s4_jobbert_haiku":       ("jobbert", "AchrafSoltani/jobbert-ner-haiku-v1"),
    "s5_jobbert_onnx_sonnet": ("onnx",    "AchrafSoltani/jobbert-ner-sonnet-v1-onnx"),
    "s6_jobbert_onnx_haiku":  ("onnx",    "AchrafSoltani/jobbert-ner-haiku-v1-onnx"),
}

LOCAL_PATHS: dict[str, tuple[str, str]] = {
    "s3_jobbert_sonnet":      ("jobbert", "pipeline/training/experiments/outputs/s3_jobbert_sonnet/checkpoint-best/checkpoint-best"),
    "s4_jobbert_haiku":       ("jobbert", "pipeline/training/experiments/outputs/s4_jobbert_haiku/checkpoint-best/checkpoint-best"),
    "s5_jobbert_onnx_sonnet": ("onnx",    "pipeline/training/experiments/outputs/s5_jobbert_onnx_sonnet/model-quantized/model-quantized"),
    "s6_jobbert_onnx_haiku":  ("onnx",    "pipeline/training/experiments/outputs/s6_jobbert_onnx_haiku/model-quantized/model-quantized"),
}

TEACHER_BY_SID: dict[str, str] = {
    "s3_jobbert_sonnet":      "sonnet",
    "s4_jobbert_haiku":       "haiku",
    "s5_jobbert_onnx_sonnet": "sonnet",
    "s6_jobbert_onnx_haiku":  "haiku",
}


def _resolve_local(rel: str) -> str:
    primary = ROOT / rel
    if primary.exists():
        return str(primary)
    # Fall back to the non-double-nested layout if the trailing duplicate is missing.
    trimmed = rel.rsplit("/", 1)[0]
    fallback = ROOT / trimmed
    return str(fallback)


def run_cell(
    sid: str,
    model_type: str,
    model_path: str,
    gold_df: pd.DataFrame,
    rng: np.random.Generator,
) -> dict:
    t0 = time.perf_counter()
    predict = build_sliding_predictor(
        model_type, model_path,
        window_tokens=WINDOW_TOKENS, stride_tokens=STRIDE_TOKENS,
    )
    pred_map: dict[str, list[dict]] = {}
    per_posting_latency_ms: list[float] = []
    for _, row in gold_df.iterrows():
        t_start = time.perf_counter()
        pred_map[row["job_link"]] = predict(row["job_summary"])
        per_posting_latency_ms.append((time.perf_counter() - t_start) * 1000)

    gold_map = dict(zip(gold_df["job_link"], gold_df["entities"]))
    contribs = per_posting_contributions(gold_map, pred_map)
    r = bootstrap_f1(contribs, N_BOOTSTRAP, rng)
    r["point_estimate"] = _point_f1(contribs)
    r["config"] = {
        "model_type": model_type,
        "model_path": model_path,
        "window_tokens": WINDOW_TOKENS,
        "stride_tokens": STRIDE_TOKENS,
    }
    sorted_lat = sorted(per_posting_latency_ms)
    r["latency_ms"] = {
        "mean": round(sum(per_posting_latency_ms) / len(per_posting_latency_ms), 2)
                if per_posting_latency_ms else 0.0,
        "p50": round(sorted_lat[len(sorted_lat) // 2], 2) if sorted_lat else 0.0,
        "p99": round(sorted_lat[int(len(sorted_lat) * 0.99)], 2) if sorted_lat else 0.0,
    }
    r["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return r


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--source", choices=["hub", "local"], default="hub")
    args = parser.parse_args()

    if args.source == "hub":
        paths = HUB_PATHS
    else:
        paths = {sid: (mt, _resolve_local(p)) for sid, (mt, p) in LOCAL_PATHS.items()}

    rng = np.random.default_rng(SEED)
    gold_s = pd.read_parquet(GOLD_SONNET)
    gold_h = pd.read_parquet(GOLD_HAIKU)
    gold_by_teacher = {"sonnet": gold_s, "haiku": gold_h}

    results: dict[str, dict] = {}
    for sid, (model_type, model_path) in paths.items():
        gold_df = gold_by_teacher[TEACHER_BY_SID[sid]]
        print(f"[{sid}] type={model_type} path={model_path}", flush=True)
        try:
            r = run_cell(sid, model_type, model_path, gold_df, rng)
            results[sid] = r
            print(
                f"  F1={r['point_estimate']:.4f}  "
                f"CI=[{r['f1_ci_lo']:.4f}, {r['f1_ci_hi']:.4f}]  "
                f"n={r['n_postings']}  "
                f"latency_mean={r['latency_ms']['mean']} ms  "
                f"elapsed={r['elapsed_s']} s",
                flush=True,
            )
        except Exception as exc:
            import traceback
            results[sid] = {"error": str(exc), "traceback": traceback.format_exc()}
            print(f"  FAILED: {exc}", flush=True)

        # Incremental write so partial results survive a mid-run crash.
        results["_meta"] = {
            "seed": SEED,
            "n_bootstrap": N_BOOTSTRAP,
            "window_tokens": WINDOW_TOKENS,
            "stride_tokens": STRIDE_TOKENS,
            "source": args.source,
            "completed": [k for k in results if k != "_meta" and "error" not in results[k]],
            "failed": [k for k in results if k != "_meta" and "error" in results[k]],
        }
        OUT.write_text(json.dumps(results, indent=2))

    print(f"\nWrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
