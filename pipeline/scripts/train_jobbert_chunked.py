"""
Chunked-window training for JobBERT NER on long job postings.

Motivation
----------
The standard `train_jobbert.py` produces ONE training example per posting with
`truncation=True, max_length=512`. Postings average ~800-1000 tokens, so the
classifier head only ever sees the first ~512 tokens of each posting at training
time. The sliding-window INFERENCE ablation (2026-04-21) showed this is a
training-time constraint, not an inference-time artefact: at inference, feeding
later chunks to the trained head produces high-recall low-precision noise
because the head was never trained on positions past 512 of a posting.

This script puts later-posting content IN-DISTRIBUTION at training time by
slicing each posting into overlapping 450-token windows (stride 225 = 50%
overlap), projecting the gold entity char-spans onto each window, and treating
each window as an independent training example. Train-time and inference-time
windowing match exactly (same window/stride, same boundary policy).

Self-contained: no imports from `train_jobbert.py` or `evaluate_student.py`.
The label list, BIO conversion, sliding logic, and end-of-training entity-set
eval are all inlined. This is intentional --- the chunked experiment is a
clean-room ablation, not an extension of the v1 training script.

Usage
-----
    python -m pipeline.scripts.train_jobbert_chunked \
        --spec pipeline/training/experiments/specs/s4_jobbert_chunked_haiku.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

ENTITY_TYPES = [
    "SKILL", "JOB_TITLE", "COMPANY", "LOCATION",
    "EXPERIENCE_LEVEL", "EDUCATION", "CERT", "COMPENSATION",
]
LABEL_LIST = ["O"] + [f"{prefix}-{t}" for t in ENTITY_TYPES for prefix in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def load_spec(spec_path: Path) -> dict:
    with spec_path.open() as f:
        return yaml.safe_load(f)


def slide_token_windows(n_tokens: int, window: int, stride: int) -> Iterable[tuple[int, int]]:
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


def chunk_to_bio_example(
    chunk_text: str,
    chunk_entities: list[dict],
    tokenizer,
    max_length: int,
) -> dict:
    """Tokenise a single chunk and produce BIO-tagged input_ids / attention_mask / labels."""
    encoding = tokenizer(
        chunk_text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        padding=False,
    )
    offsets = encoding["offset_mapping"]
    special_mask = encoding["special_tokens_mask"]

    char_labels = ["O"] * len(chunk_text)
    sorted_ents = sorted(chunk_entities, key=lambda e: (e["start"], -(e["end"] - e["start"])))
    for ent in sorted_ents:
        s, e, t = ent["start"], ent["end"], ent["type"]
        if s < 0 or e > len(chunk_text) or s >= e:
            continue
        if all(char_labels[i] == "O" for i in range(s, e)):
            char_labels[s] = f"B-{t}"
            for i in range(s + 1, e):
                char_labels[i] = f"I-{t}"

    token_labels = []
    for offset, is_special in zip(offsets, special_mask):
        if is_special or offset[0] == offset[1]:
            token_labels.append(-100)
        else:
            label_str = char_labels[offset[0]] if offset[0] < len(char_labels) else "O"
            token_labels.append(LABEL2ID.get(label_str, 0))

    encoding["labels"] = token_labels
    del encoding["offset_mapping"]
    del encoding["special_tokens_mask"]
    return encoding


def parquet_to_chunked_dataset(
    parquet_path: Path,
    tokenizer,
    window_tokens: int,
    stride_tokens: int,
    max_length: int,
):
    """Slice each posting into overlapping windows; one training example per window."""
    from datasets import Dataset

    df = pd.read_parquet(parquet_path)
    examples: list[dict] = []
    n_postings = 0
    n_entities_total = 0
    n_entities_kept = 0
    n_boundary_overlap_events = 0
    chunks_per_posting: list[int] = []

    for _, row in df.iterrows():
        text = str(row["job_summary"])[:20_000]
        entities = row["entities"]
        if isinstance(entities, np.ndarray):
            entities = entities.tolist()
        if not isinstance(entities, list):
            entities = []
        n_entities_total += len(entities)
        n_postings += 1

        full_enc = tokenizer(
            text, truncation=False,
            return_offsets_mapping=True, add_special_tokens=False,
        )
        full_offsets = full_enc["offset_mapping"]
        n_tokens = len(full_offsets)
        if n_tokens == 0:
            chunks_per_posting.append(0)
            continue

        posting_chunks = 0
        for start_tok, end_tok in slide_token_windows(n_tokens, window_tokens, stride_tokens):
            char_start = full_offsets[start_tok][0]
            char_end = full_offsets[end_tok - 1][1]
            chunk = text[char_start:char_end]
            if not chunk.strip():
                continue

            projected = []
            for ent in entities:
                e_start = int(ent["start"])
                e_end = int(ent["end"])
                if e_start >= char_start and e_end <= char_end:
                    projected.append({
                        "start": e_start - char_start,
                        "end": e_end - char_start,
                        "type": ent["type"],
                    })
                elif not (e_end <= char_start or e_start >= char_end):
                    n_boundary_overlap_events += 1

            n_entities_kept += len(projected)
            ex = chunk_to_bio_example(chunk, projected, tokenizer, max_length)
            examples.append(ex)
            posting_chunks += 1
        chunks_per_posting.append(posting_chunks)

    avg_chunks = sum(chunks_per_posting) / max(1, n_postings)
    logger.info(
        f"Parquet {parquet_path.name}: {n_postings} postings -> {len(examples)} chunked examples "
        f"(avg {avg_chunks:.2f} chunks/posting; "
        f"{n_entities_total} entities, {n_entities_kept} kept-in-window, "
        f"{n_boundary_overlap_events} boundary-overlap events)"
    )
    return Dataset.from_dict({
        "input_ids": [r["input_ids"] for r in examples],
        "attention_mask": [r["attention_mask"] for r in examples],
        "labels": [r["labels"] for r in examples],
    })


def build_sliding_predictor(
    model_path: str, tokenizer, window_tokens: int, stride_tokens: int,
) -> Callable[[str], list[dict]]:
    """Sliding-window inference matching the train-time windowing."""
    from transformers import AutoModelForTokenClassification, pipeline as hf_pipeline
    import torch

    mdl = AutoModelForTokenClassification.from_pretrained(model_path)
    device = 0 if torch.cuda.is_available() else -1
    pipe = hf_pipeline(
        "token-classification", model=mdl, tokenizer=tokenizer,
        aggregation_strategy="simple", device=device,
    )

    def predict(text: str) -> list[dict]:
        text = text[:20_000]
        if not text.strip():
            return []
        enc = tokenizer(text, return_offsets_mapping=True,
                        add_special_tokens=False, truncation=False)
        offsets = enc["offset_mapping"]
        n = len(offsets)
        if n == 0:
            return []
        seen: set[tuple[str, int, int]] = set()
        out: list[dict] = []
        for s_tok, e_tok in slide_token_windows(n, window_tokens, stride_tokens):
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


def entity_set_eval(gold_df: pd.DataFrame, predict_fn, split: str) -> dict:
    """Inline entity-set eval --- comparable to teacher/student v1 numbers."""
    tp_c, fp_c, fn_c = Counter(), Counter(), Counter()
    latencies: list[float] = []
    for _, row in gold_df.iterrows():
        gold = row["entities"]
        if hasattr(gold, "tolist"):
            gold = gold.tolist()
        gold_set = {(e["text"], e["type"]) for e in gold if e.get("text") and e.get("type")}

        t0 = time.perf_counter()
        pred = predict_fn(row["job_summary"])
        latencies.append((time.perf_counter() - t0) * 1000)

        pred_set = {(e["text"], e["type"]) for e in pred if e.get("text") and e.get("type")}
        for e in gold_set & pred_set:
            tp_c[e[1]] += 1
        for e in pred_set - gold_set:
            fp_c[e[1]] += 1
        for e in gold_set - pred_set:
            fn_c[e[1]] += 1

    tp, fp, fn = sum(tp_c.values()), sum(fp_c.values()), sum(fn_c.values())
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0

    per_type = {}
    for t in sorted(set(tp_c) | set(fp_c) | set(fn_c)):
        tpv, fpv, fnv = tp_c[t], fp_c[t], fn_c[t]
        pp = tpv / (tpv + fpv) if tpv + fpv > 0 else 0.0
        rr = tpv / (tpv + fnv) if tpv + fnv > 0 else 0.0
        ff = 2 * pp * rr / (pp + rr) if pp + rr > 0 else 0.0
        per_type[t] = {"tp": tpv, "fp": fpv, "fn": fnv,
                       "precision": round(pp, 4), "recall": round(rr, 4), "f1": round(ff, 4)}

    avg_lat = sum(latencies) / max(1, len(latencies))
    return {
        "split": split,
        "postings": len(gold_df),
        "micro": {"tp": tp, "fp": fp, "fn": fn,
                  "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)},
        "per_type": per_type,
        "latency_ms_mean": round(avg_lat, 2),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Chunked-window JobBERT NER training")
    parser.add_argument("--spec", type=Path, required=True, help="Experiment spec YAML")
    parser.add_argument("--skip-hub", action="store_true",
                        help="Force-skip Hub push regardless of spec")
    args = parser.parse_args()

    spec = load_spec(args.spec)
    exp = spec["experiment"]
    hp = spec["hyperparameters"]
    chunking = spec.get("chunking", {})
    window_tokens = chunking.get("window_tokens", 450)
    stride_tokens = chunking.get("stride_tokens", 225)
    output_dir = REPO_ROOT / "pipeline" / "training" / "experiments" / spec["output"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== {exp['id']} (chunked window={window_tokens} stride={stride_tokens}) ===")

    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )
    import evaluate as hf_evaluate

    tokenizer = AutoTokenizer.from_pretrained(spec["model"]["base"])
    max_len = hp.get("max_seq_length", 512)

    logger.info("Building chunked train set...")
    train_ds = parquet_to_chunked_dataset(
        REPO_ROOT / spec["data"]["train"], tokenizer,
        window_tokens, stride_tokens, max_len,
    )
    logger.info("Building chunked dev set...")
    dev_ds = parquet_to_chunked_dataset(
        REPO_ROOT / spec["data"]["dev"], tokenizer,
        window_tokens, stride_tokens, max_len,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        spec["model"]["base"],
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    seqeval = hf_evaluate.load("seqeval")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        preds = np.argmax(logits, axis=-1)
        true_l, true_p = [], []
        for ps, ls in zip(preds, labels):
            tl, tp_ = [], []
            for p, l in zip(ps, ls):
                if l != -100:
                    tl.append(ID2LABEL[l])
                    tp_.append(ID2LABEL[p])
            true_l.append(tl)
            true_p.append(tp_)
        res = seqeval.compute(predictions=true_p, references=true_l)
        return {
            "precision": res["overall_precision"],
            "recall": res["overall_recall"],
            "f1": res["overall_f1"],
            "accuracy": res["overall_accuracy"],
        }

    collator = DataCollatorForTokenClassification(tokenizer, padding=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=hp["epochs"],
        per_device_train_batch_size=hp["per_device_train_batch_size"],
        per_device_eval_batch_size=hp["per_device_eval_batch_size"],
        learning_rate=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
        warmup_ratio=hp["warmup_ratio"],
        eval_strategy=hp["eval_strategy"],
        save_strategy=hp["save_strategy"],
        load_best_model_at_end=hp["load_best_model_at_end"],
        metric_for_best_model=hp["metric_for_best_model"],
        fp16=hp.get("fp16", True),
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=dev_ds,
        processing_class=tokenizer, data_collator=collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    best_dir = output_dir / "checkpoint-best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    logger.info(f"Saved best model to {best_dir}")

    eval_results: dict = {"chunking": {"window_tokens": window_tokens,
                                       "stride_tokens": stride_tokens}}
    gold_path = REPO_ROOT / spec["data"]["gold"]
    if gold_path.exists():
        logger.info("Sliding-window entity-set eval on gold (matches inference-time conditions)...")
        predict = build_sliding_predictor(str(best_dir), tokenizer,
                                          window_tokens, stride_tokens)
        gold_df = pd.read_parquet(gold_path)
        eval_results["gold_sliding"] = entity_set_eval(gold_df, predict, "gold_sliding")
        m = eval_results["gold_sliding"]["micro"]
        logger.info(f"Gold sliding micro P/R/F1: {m['precision']:.4f} / {m['recall']:.4f} / {m['f1']:.4f}")

    eval_path = output_dir / "eval.json"
    with eval_path.open("w") as f:
        json.dump(eval_results, f, indent=2)
    logger.info(f"Wrote {eval_path}")

    if spec["output"].get("push_to_hub") and not args.skip_hub:
        hub_id = exp["hub_model_id"]
        logger.info(f"Pushing to Hub: {hub_id}")
        trainer.push_to_hub(hub_id)
        logger.info(f"Pushed to {hub_id}")
    else:
        logger.info("Skipping Hub push (push_to_hub=false or --skip-hub).")

    logger.info(f"=== {exp['id']} COMPLETE ===")


if __name__ == "__main__":
    main()
