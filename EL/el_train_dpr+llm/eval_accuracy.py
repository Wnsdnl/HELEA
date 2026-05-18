"""
이진 분류 평가: DPR 점수 + LLM 점수 융합.

reranking이 아님 — 미리 정해진 (entity_a, entity_b, label) 쌍에 대한 분류.

각 쌍에 대해:
  1. DPR: 두 entity의 cosine 유사도 계산 (dpr_score)
  2. LLM: 독립적으로 점수 부여 (llm_score, 0~1)
  3. final_score = alpha * dpr_score + (1 - alpha) * llm_scaled
  4. threshold sweep → 최적 threshold → binary 예측 → best F1

데이터: DW_1H_29k_EN.jsonl (29K 쌍, label 0/1 혼합)
"""

import json
import argparse
import sys
import numpy as np
import torch
import wandb
from transformers import AutoTokenizer
from torch.amp import autocast

import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "Utils/llm"))
from core.async_vllm_client import AsyncVLLMClient

sys.path.append(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "el_train_dpr"))
from model import BiEncoderModel
from dataset import serialize_entity

from cfg import Config
from prompt_builder import build_reranking_prompt, parse_response, RERANKING_SYSTEM_PROMPT
from reranker import fuse_scores


def load_acc_data(path: str, max_hops: int):
    names_a, names_b = [], []
    hops_a_list, hops_b_list = [], []
    texts_a, texts_b = [], []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ea = d.get("entity_a", "")
            eb = d.get("entity_b", "")
            ha = d.get("1-hops_a", [])
            hb = d.get("1-hops_b", [])

            names_a.append(ea)
            names_b.append(eb)
            hops_a_list.append(ha)
            hops_b_list.append(hb)
            texts_a.append(serialize_entity(ha, ea, max_hops))
            texts_b.append(serialize_entity(hb, eb, max_hops))
            labels.append(float(d.get("label", 0.0)))

    return names_a, names_b, hops_a_list, hops_b_list, texts_a, texts_b, labels


@torch.no_grad()
def compute_dpr_scores(
    texts_a: list,
    texts_b: list,
    model,
    tokenizer,
    device: torch.device,
    cfg: Config,
) -> np.ndarray:
    """Pairwise cosine similarity for all pairs. Returns (N,) float32 array."""
    batch_size = cfg.index_batch_size
    all_sims = []
    use_cuda = torch.cuda.is_available()
    ctx = autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda)

    for i in range(0, len(texts_a), batch_size):
        ba = texts_a[i:i + batch_size]
        bb = texts_b[i:i + batch_size]

        tok_a = tokenizer(ba, padding="max_length", truncation=True,
                          max_length=cfg.max_length, return_tensors="pt")
        tok_b = tokenizer(bb, padding="max_length", truncation=True,
                          max_length=cfg.max_length, return_tensors="pt")

        ids_a  = tok_a["input_ids"].to(device)
        mask_a = tok_a["attention_mask"].to(device)
        ids_b  = tok_b["input_ids"].to(device)
        mask_b = tok_b["attention_mask"].to(device)

        with ctx:
            emb_a = model(ids_a, mask_a)
            emb_b = model(ids_b, mask_b)
            sims  = torch.sum(emb_a * emb_b, dim=1)

        all_sims.extend(sims.cpu().tolist())

        if i > 0 and i % (batch_size * 10) == 0:
            print(f"  DPR encoded {i}/{len(texts_a)}")

    return np.array(all_sims, dtype=np.float32)


def compute_llm_scores(
    names_a: list,
    hops_a_list: list,
    names_b: list,
    hops_b_list: list,
    cfg: Config,
) -> np.ndarray:
    """Pointwise LLM scoring for all pairs. Returns (N,) float32 array in [0, 1]."""
    prompts = [
        build_reranking_prompt(
            entity_a=names_a[i],
            hops_a=hops_a_list[i],
            entity_b=names_b[i],
            hops_b=hops_b_list[i],
            max_triples=cfg.max_triples,
        )
        for i in range(len(names_a))
    ]

    print(f"  Sending {len(prompts)} LLM requests (concurrency={cfg.concurrency})...")
    client = AsyncVLLMClient(cfg)
    raw_responses = client.run(
        prompts,
        system_prompt=RERANKING_SYSTEM_PROMPT,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )

    scores = [parse_response(r)["score"] for r in raw_responses]
    return np.array(scores, dtype=np.float32)


def threshold_sweep(final_scores: np.ndarray, labels: np.ndarray) -> dict:
    """Sweep thresholds in [-1, 1], return best threshold and metrics."""
    best_f1 = 0.0
    best_threshold = 0.0
    thresholds = np.arange(-1.0, 1.0, 0.02)

    for th in thresholds:
        preds = (final_scores >= th).astype(float)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        denom = 2 * tp + fp + fn
        f1 = (2 * tp) / denom if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th

    best_preds = (final_scores >= best_threshold).astype(float)
    tp = np.sum((best_preds == 1) & (labels == 1))
    fp = np.sum((best_preds == 1) & (labels == 0))
    fn = np.sum((best_preds == 0) & (labels == 1))
    tn = np.sum((best_preds == 0) & (labels == 0))

    acc       = float(np.mean(best_preds == labels))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall    = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1        = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "best_threshold": float(best_threshold),
        "accuracy":       acc,
        "precision":      precision,
        "recall":         recall,
        "f1":             f1,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


@torch.no_grad()
def evaluate_accuracy(cfg: Config, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Support --alphas "1.0,0.75,0.5,0.25,0.0" for DPR+LLM computed once, alpha swept.
    alphas_str = getattr(args, "alphas", None)
    if alphas_str:
        alphas = [float(a.strip()) for a in alphas_str.split(",")]
    else:
        alphas = [cfg.acc_alpha]

    wandb.init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        name=f"DPR+LLM-Accuracy-alphasweep",
        config={
            "dpr_checkpoint": cfg.dpr_checkpoint,
            "alphas":         alphas,
            "max_hops":       cfg.max_hops,
        },
    )

    print(f"Loading model from: {cfg.dpr_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.dpr_checkpoint, use_fast=False)
    model = BiEncoderModel(cfg.dpr_checkpoint).to(device)
    model.eval()

    print(f"Loading data from: {cfg.acc_eval_path}")
    names_a, names_b, hops_a_list, hops_b_list, texts_a, texts_b, labels = load_acc_data(
        cfg.acc_eval_path, cfg.max_hops
    )
    labels = np.array(labels)
    N = len(labels)
    print(f"Loaded {N} pairs  (pos={int(labels.sum())}, neg={int((1-labels).sum())})")

    print(f"\nComputing DPR pairwise similarities...")
    dpr_scores = compute_dpr_scores(texts_a, texts_b, model, tokenizer, device, cfg)

    print(f"\nComputing LLM scores (once, reused for all alphas)...")
    llm_scores = compute_llm_scores(names_a, hops_a_list, names_b, hops_b_list, cfg)

    sims_pos = dpr_scores[labels == 1]
    sims_neg = dpr_scores[labels == 0]
    if len(sims_pos) > 0:
        print(f"Avg DPR sim (label=1): {np.mean(sims_pos):.4f}")
    if len(sims_neg) > 0:
        print(f"Avg DPR sim (label=0): {np.mean(sims_neg):.4f}")

    # Sweep all alphas without re-calling DPR or LLM
    all_metrics = {}
    for alpha in alphas:
        final_scores = alpha * dpr_scores + (1.0 - alpha) * llm_scores
        metrics = threshold_sweep(final_scores, labels)
        all_metrics[alpha] = metrics

        print(f"\n===== α={alpha} — DPR+LLM Score Fusion =====")
        print(f"Best Threshold : {metrics['best_threshold']:.2f}")
        print(f"Accuracy       : {metrics['accuracy']  * 100:.2f}%")
        print(f"Precision      : {metrics['precision'] * 100:.2f}%")
        print(f"Recall         : {metrics['recall']    * 100:.2f}%")
        print(f"F1 Score       : {metrics['f1']        * 100:.2f}%")
        print(f"TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']}")
        wandb.log({f"alpha_{alpha}/{k}": v for k, v in metrics.items()})

    wandb.finish()
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha",        type=float, default=None,
                        help="단일 alpha 값 (기존 호환용)")
    parser.add_argument("--alphas",       type=str,   default=None,
                        help="콤마 구분 alpha 목록, DPR+LLM 각 1회 후 모두 계산 (예: '1.0,0.75,0.5,0.25,0.0')")
    parser.add_argument("--checkpoint",   type=str,   default=None)
    parser.add_argument("--acc_eval_path", type=str,  default=None)
    parser.add_argument("--llm_url",      type=str,   default=None,
                        help="vLLM gateway URL (예: http://localhost:8004)")
    parser.add_argument("--model",        type=str,   default=None,
                        help="모델명 (예: openai/gpt-oss-120b)")
    args = parser.parse_args()

    cfg = Config()
    if args.alpha is not None:
        cfg.acc_alpha = args.alpha
    if args.checkpoint is not None:
        cfg.dpr_checkpoint = args.checkpoint
    if args.acc_eval_path is not None:
        cfg.acc_eval_path = args.acc_eval_path
    if args.llm_url is not None:
        cfg.llm_url = args.llm_url
    if args.model is not None:
        cfg.model = args.model

    evaluate_accuracy(cfg, args)
