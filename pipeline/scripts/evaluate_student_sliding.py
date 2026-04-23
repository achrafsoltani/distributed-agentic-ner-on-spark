"""
Sliding-window inference for JobBERT and ONNX students.

Motivation
----------
JobBERT (BERT-base, 512 positional embeddings) and its ONNX int8 variant can only
attend to 512 tokens per forward pass. Our LinkedIn postings average ~4,000
characters (~800-1000 tokens), so single-shot inference via the Hugging Face
pipeline truncates everything past position 512 --- roughly the last two-thirds
of each posting. This is a coverage artefact of the inference loop, not a
model-capacity limit: the same architecture can see the full posting via a
sliding window over token positions. This module provides that path.

Window strategy
---------------
- Tokenise the full text without truncation to obtain (char_start, char_end)
  offsets per token.
- Slide a window of `window_tokens` (default 450, leaving 62-token headroom
  under the 512 limit for [CLS]/[SEP] and subword re-expansion) with
  `stride_tokens` (default 225 = 50% overlap).
- For each window, slice the original text between the char offsets of the
  first and last token in the window, and hand that slice to the HF
  token-classification pipeline.
- Shift pipeline-reported entity char offsets back to the original text frame
  and deduplicate on (type, char_start, char_end).
- Boundary suppression: drop entities that touch a window's inner edge (not
  also the edge of the full text). Such entities risk being subword-truncated
  at the window boundary and will be captured intact in a neighbouring window
  given the 50% overlap.

Model source
------------
`--model-path` accepts either a local filesystem path (the Phase 6 output
directory) or a Hugging Face Hub ID (e.g. `AchrafSoltani/jobbert-ner-haiku-v1`).
The AWS re-run uses Hub paths to avoid staging model weights on S3.

Usage
-----
    # Single student via Hub
    python -m pipeline.scripts.evaluate_student_sliding \
        --model-type jobbert \
        --model-path AchrafSoltani/jobbert-ner-haiku-v1 \
        --gold pipeline/training/haiku/gold.parquet \
        --output pipeline/training/sliding/s4_jobbert_haiku.json

    # ONNX variant, explicit window/stride
    python -m pipeline.scripts.evaluate_student_sliding \
        --model-type onnx \
        --model-path AchrafSoltani/jobbert-ner-haiku-v1-onnx \
        --gold pipeline/training/haiku/gold.parquet \
        --output pipeline/training/sliding/s6_onnx_haiku.json \
        --window-tokens 450 --stride-tokens 225
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

logger = logging.getLogger(__name__)


def _slide_windows(n_tokens: int, window: int, stride: int) -> Iterable[tuple[int, int]]:
    """Yield (start_tok, end_tok) pairs covering [0, n_tokens) with window/stride."""
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


def _jobbert_pipeline(model_path: str):
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline as hf_pipeline
    import torch
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForTokenClassification.from_pretrained(model_path)
    device = 0 if torch.cuda.is_available() else -1
    pipe = hf_pipeline(
        "token-classification", model=mdl, tokenizer=tok,
        aggregation_strategy="simple", device=device,
    )
    return tok, pipe


def _onnx_pipeline(model_path: str, file_name: str = "model_quantized.onnx"):
    from optimum.onnxruntime import ORTModelForTokenClassification
    from transformers import AutoTokenizer, pipeline as hf_pipeline
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = ORTModelForTokenClassification.from_pretrained(model_path, file_name=file_name)
    # ONNX int8 models are shipped as CPU-deployment artefacts (int8 CUDA kernels
    # are not universally available). Force CPU explicitly so a GPU host does
    # not try to move the model onto CUDAExecutionProvider.
    pipe = hf_pipeline(
        "token-classification", model=mdl, tokenizer=tok,
        aggregation_strategy="simple", device=-1,
    )
    return tok, pipe


def build_sliding_predictor(
    model_type: str,
    model_path: str,
    window_tokens: int = 450,
    stride_tokens: int = 225,
    onnx_file_name: str = "model_quantized.onnx",
) -> Callable[[str], list[dict]]:
    """Return a predict(text) function that applies sliding-window inference."""
    if model_type == "jobbert":
        tok, pipe = _jobbert_pipeline(model_path)
    elif model_type == "onnx":
        tok, pipe = _onnx_pipeline(model_path, file_name=onnx_file_name)
    else:
        raise ValueError(f"Unsupported model_type for sliding windows: {model_type}")

    def predict(text: str) -> list[dict]:
        text = text[:20_000]
        if not text.strip():
            return []
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False, truncation=False)
        offsets = enc["offset_mapping"]
        n_tokens = len(offsets)
        if n_tokens == 0:
            return []

        seen: set[tuple[str, int, int]] = set()
        entities: list[dict] = []
        for start_tok, end_tok in _slide_windows(n_tokens, window_tokens, stride_tokens):
            char_start = offsets[start_tok][0]
            char_end = offsets[end_tok - 1][1]
            chunk = text[char_start:char_end]
            if not chunk.strip():
                continue

            results = pipe(chunk)
            inner_left = start_tok > 0
            inner_right = end_tok < n_tokens
            chunk_len = len(chunk)
            for r in results:
                touches_left = r["start"] <= 1 and inner_left
                touches_right = r["end"] >= chunk_len - 1 and inner_right
                if touches_left or touches_right:
                    continue
                g_start = char_start + int(r["start"])
                g_end = char_start + int(r["end"])
                key = (r["entity_group"], g_start, g_end)
                if key in seen:
                    continue
                seen.add(key)
                entities.append({
                    "text": r["word"],
                    "type": r["entity_group"],
                    "start": g_start,
                    "end": g_end,
                })
        return entities

    return predict


def evaluate(gold_df: pd.DataFrame, predict_fn: Callable[[str], list[dict]], split: str = "gold") -> dict:
    tp_c, fp_c, fn_c = Counter(), Counter(), Counter()
    latencies_ms = []
    processed = 0
    for _, row in gold_df.iterrows():
        text = row["job_summary"]
        gold_ents = row["entities"]
        if hasattr(gold_ents, "tolist"):
            gold_ents = gold_ents.tolist()
        gold_set = {(e["text"], e["type"]) for e in gold_ents if e.get("text") and e.get("type")}

        t0 = time.perf_counter()
        pred_ents = predict_fn(text)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)

        pred_set = {(e["text"], e["type"]) for e in pred_ents if e.get("text") and e.get("type")}
        for e in gold_set & pred_set:
            tp_c[e[1]] += 1
        for e in pred_set - gold_set:
            fp_c[e[1]] += 1
        for e in gold_set - pred_set:
            fn_c[e[1]] += 1
        processed += 1

    total_tp = sum(tp_c.values())
    total_fp = sum(fp_c.values())
    total_fn = sum(fn_c.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    per_type = {}
    for t in sorted(set(tp_c) | set(fp_c) | set(fn_c)):
        tp_v, fp_v, fn_v = tp_c[t], fp_c[t], fn_c[t]
        p = tp_v / (tp_v + fp_v) if (tp_v + fp_v) > 0 else 0.0
        r = tp_v / (tp_v + fn_v) if (tp_v + fn_v) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_type[t] = {
            "tp": tp_v, "fp": fp_v, "fn": fn_v,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        }

    avg = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    sorted_lat = sorted(latencies_ms)
    p50 = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0.0
    p99 = sorted_lat[int(len(sorted_lat) * 0.99)] if sorted_lat else 0.0

    return {
        "split": split,
        "postings_evaluated": processed,
        "micro": {
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "precision": round(micro_p, 4),
            "recall": round(micro_r, 4),
            "f1": round(micro_f1, 4),
        },
        "per_type": per_type,
        "latency_ms": {"mean": round(avg, 2), "p50": round(p50, 2), "p99": round(p99, 2)},
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Sliding-window evaluation for JobBERT / ONNX students")
    parser.add_argument("--model-type", choices=["jobbert", "onnx"], required=True)
    parser.add_argument("--model-path", required=True,
                        help="Local directory or Hugging Face Hub ID")
    parser.add_argument("--gold", type=Path, required=True, help="Gold parquet path")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--window-tokens", type=int, default=450)
    parser.add_argument("--stride-tokens", type=int, default=225)
    parser.add_argument("--onnx-file-name", default="model_quantized.onnx")
    args = parser.parse_args()

    logger.info(f"Loading {args.model_type} from {args.model_path}")
    predict = build_sliding_predictor(
        args.model_type, args.model_path,
        window_tokens=args.window_tokens,
        stride_tokens=args.stride_tokens,
        onnx_file_name=args.onnx_file_name,
    )

    logger.info(f"Evaluating on {args.gold}")
    gold_df = pd.read_parquet(args.gold)
    results = evaluate(gold_df, predict, split=args.gold.stem)
    results["config"] = {
        "model_type": args.model_type,
        "model_path": args.model_path,
        "window_tokens": args.window_tokens,
        "stride_tokens": args.stride_tokens,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(results, f, indent=2)

    m = results["micro"]
    lat = results["latency_ms"]
    logger.info(f"Micro P/R/F1: {m['precision']:.3f} / {m['recall']:.3f} / {m['f1']:.3f}")
    logger.info(f"Latency: mean={lat['mean']:.1f}ms  p50={lat['p50']:.1f}ms  p99={lat['p99']:.1f}ms")
    logger.info(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
