"""
Posting-level bootstrap CI for the two chunked-trained JobBERT students
(S3' Sonnet, S4' Haiku) evaluated with sliding-window inference on the
516-posting gold set.

Companion to bootstrap_ci.py / bootstrap_ci_sliding.py. Same SEED (42),
same N_BOOTSTRAP (10,000), same posting-level resampling --- only the
predictor and the model checkpoints change. Use the three outputs together
to fill the chunked-training row block in paper Table 5.

Self-contained: the sliding predictor and the bootstrap statistics are
inlined here rather than imported from the v1 evaluation modules. This
matches the clean-room pattern set by `train_jobbert_chunked.py` --- the
chunked experiment owns its own end-to-end measurement code.

Sources
-------
--source local (default): the two checkpoints written by
    `train_jobbert_chunked.py` under
    pipeline/training/experiments/outputs/s{3,4}_jobbert_chunked_*/
    checkpoint-best/.
--source hub:             pulls the (yet-to-be-published) v2 repos
    AchrafSoltani/jobbert-ner-{sonnet,haiku}-v2.

Usage
-----
    python -m pipeline.scripts.bootstrap_ci_chunked --source local

Writes pipeline/training/bootstrap_ci_chunked.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GOLD_SONNET = ROOT / "pipeline/training/sonnet/gold.parquet"
GOLD_HAIKU = ROOT / "pipeline/training/haiku/gold.parquet"
OUT = ROOT / "pipeline/training/bootstrap_ci_chunked.json"

N_BOOTSTRAP = 10_000
SEED = 42
WINDOW_TOKENS = 450
STRIDE_TOKENS = 225

LOCAL_PATHS: dict[str, str] = {
    "s3_jobbert_chunked_sonnet": "pipeline/training/experiments/outputs/s3_jobbert_chunked_sonnet/checkpoint-best",
    "s4_jobbert_chunked_haiku":  "pipeline/training/experiments/outputs/s4_jobbert_chunked_haiku/checkpoint-best",
}

HUB_PATHS: dict[str, str] = {
    "s3_jobbert_chunked_sonnet": "AchrafSoltani/jobbert-ner-sonnet-v2",
    "s4_jobbert_chunked_haiku":  "AchrafSoltani/jobbert-ner-haiku-v2",
}

TEACHER_BY_SID: dict[str, str] = {
    "s3_jobbert_chunked_sonnet": "sonnet",
    "s4_jobbert_chunked_haiku":  "haiku",
}


def slide_token_windows(n_tokens: int, window: int, stride: int) -> Iterable[tuple[int, int]]:
    if n_tokens <= window:
        yield 0, n_tokens
        return
    start = 0
    while start < n_tokens:
        end = min(start + window, n_tokens)
        yield start, end
        if end == n_tokens:
            break
        start += stride


def build_chunked_sliding_predictor(model_path: str) -> Callable[[str], list[dict]]:
    """Sliding-window inference matching train_jobbert_chunked.py exactly."""
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline as hf_pipeline
    import torch

    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForTokenClassification.from_pretrained(model_path)
    device = 0 if torch.cuda.is_available() else -1
    pipe = hf_pipeline(
        "token-classification", model=mdl, tokenizer=tok,
        aggregation_strategy="simple", device=device,
    )

    def predict(text: str) -> list[dict]:
        text = text[:20_000]
        if not text.strip():
            return []
        enc = tok(text, return_offsets_mapping=True,
                  add_special_tokens=False, truncation=False)
        offsets = enc["offset_mapping"]
        n = len(offsets)
        if n == 0:
            return []
        seen: set[tuple[str, int, int]] = set()
        out: list[dict] = []
        for s_tok, e_tok in slide_token_windows(n, WINDOW_TOKENS, STRIDE_TOKENS):
            cs = offsets[s_tok][0]
            ce = offsets[e_tok - 1][1]
            chunk = text[cs:ce]
            if not chunk.strip():
                continue
            inner_left = s_tok > 0
            inner_right = e_tok < n
            chunk_len = len(chunk)
            for r in pipe(chunk):
                touches_left = r["start"] <= 1 and inner_left
                touches_right = r["end"] >= chunk_len - 1 and inner_right
                if touches_left or touches_right:
                    continue
                gs = cs + int(r["start"])
                ge = cs + int(r["end"])
                key = (r["entity_group"], gs, ge)
                if key in seen:
                    continue
                seen.add(key)
                out.append({
                    "text": r["word"], "type": r["entity_group"],
                    "start": gs, "end": ge,
                })
        return out

    return predict


def entity_set(entities) -> set[tuple[str, str]]:
    if entities is None:
        return set()
    if hasattr(entities, "tolist"):
        entities = entities.tolist()
    out: set[tuple[str, str]] = set()
    for e in entities:
        if isinstance(e, dict) and e.get("text") and e.get("type"):
            out.add((e["text"], e["type"]))
    return out


def per_posting_contributions(
    gold_map: dict, pred_map: dict,
) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    for link, gold_ents in gold_map.items():
        gold_s = entity_set(gold_ents)
        pred_s = entity_set(pred_map.get(link, []))
        tp = len(gold_s & pred_s)
        fp = len(pred_s - gold_s)
        fn = len(gold_s - pred_s)
        rows.append((tp, fp, fn))
    return rows


def point_f1(contribs: list[tuple[int, int, int]]) -> float:
    tp = sum(c[0] for c in contribs)
    fp = sum(c[1] for c in contribs)
    fn = sum(c[2] for c in contribs)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0


def bootstrap_f1(
    contribs: list[tuple[int, int, int]], n_boot: int, rng: np.random.Generator,
) -> dict:
    arr = np.array(contribs)
    n = len(arr)
    idx = rng.integers(0, n, size=(n_boot, n))
    sampled = arr[idx]
    tp = sampled[:, :, 0].sum(axis=1)
    fp = sampled[:, :, 1].sum(axis=1)
    fn = sampled[:, :, 2].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        rec = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    return {
        "f1_mean": float(f1.mean()),
        "f1_sd": float(f1.std(ddof=1)),
        "f1_ci_lo": float(np.percentile(f1, 2.5)),
        "f1_ci_hi": float(np.percentile(f1, 97.5)),
        "prec_mean": float(prec.mean()),
        "rec_mean": float(rec.mean()),
        "n_postings": n,
    }


def run_cell(
    sid: str, model_path: str, gold_df: pd.DataFrame, rng: np.random.Generator,
) -> dict:
    t0 = time.perf_counter()
    predict = build_chunked_sliding_predictor(model_path)

    pred_map: dict[str, list[dict]] = {}
    latencies_ms: list[float] = []
    for _, row in gold_df.iterrows():
        t_start = time.perf_counter()
        pred_map[row["job_link"]] = predict(row["job_summary"])
        latencies_ms.append((time.perf_counter() - t_start) * 1000)

    gold_map = dict(zip(gold_df["job_link"], gold_df["entities"]))
    contribs = per_posting_contributions(gold_map, pred_map)
    r = bootstrap_f1(contribs, N_BOOTSTRAP, rng)
    r["point_estimate"] = point_f1(contribs)
    r["config"] = {
        "model_path": model_path,
        "window_tokens": WINDOW_TOKENS,
        "stride_tokens": STRIDE_TOKENS,
    }
    sorted_lat = sorted(latencies_ms)
    r["latency_ms"] = {
        "mean": round(sum(latencies_ms) / len(latencies_ms), 2) if latencies_ms else 0.0,
        "p50": round(sorted_lat[len(sorted_lat) // 2], 2) if sorted_lat else 0.0,
        "p99": round(sorted_lat[int(len(sorted_lat) * 0.99)], 2) if sorted_lat else 0.0,
    }
    r["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return r


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap CI for chunked-trained JobBERT students")
    parser.add_argument("--source", choices=["local", "hub"], default="local")
    args = parser.parse_args()

    paths = LOCAL_PATHS if args.source == "local" else HUB_PATHS
    if args.source == "local":
        paths = {sid: str(ROOT / rel) for sid, rel in paths.items()}

    rng = np.random.default_rng(SEED)
    gold_s = pd.read_parquet(GOLD_SONNET)
    gold_h = pd.read_parquet(GOLD_HAIKU)
    gold_by_teacher = {"sonnet": gold_s, "haiku": gold_h}

    results: dict[str, dict] = {}
    for sid, model_path in paths.items():
        gold_df = gold_by_teacher[TEACHER_BY_SID[sid]]
        print(f"[{sid}] path={model_path}", flush=True)
        try:
            r = run_cell(sid, model_path, gold_df, rng)
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

        results["_meta"] = {
            "seed": SEED,
            "n_bootstrap": N_BOOTSTRAP,
            "window_tokens": WINDOW_TOKENS,
            "stride_tokens": STRIDE_TOKENS,
            "source": args.source,
            "completed": [k for k in results if k != "_meta" and "error" not in results[k]],
            "failed":    [k for k in results if k != "_meta" and "error" in results[k]],
        }
        OUT.write_text(json.dumps(results, indent=2))

    print(f"\nWrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
