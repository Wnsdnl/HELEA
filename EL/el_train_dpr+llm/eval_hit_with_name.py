"""
Hit@1 / Hit@5 평가 (with-name): DPR 검색 + LLM listwise 재정렬.
entity 이름을 DPR 직렬화 및 LLM 프롬프트에 모두 포함하여 name bias 비교 실험.
"""

import json
import argparse
import numpy as np
import torch
import wandb
from transformers import AutoTokenizer

import sys
import os as _os
sys.path.append(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "el_train_dpr"))
from model import BiEncoderModel
from dataset import serialize_entity_with_name

from cfg import Config
from indexer import encode_corpus, build_index, retrieve_top_k
from reranker import rerank_queries_listwise as rerank_queries


def load_hit_data(path: str, max_hops: int):
    names_a, names_b = [], []
    hops_a_list, hops_b_list = [], []
    texts_a, texts_b = [], []

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
            texts_a.append(serialize_entity_with_name(ha, ea, max_hops))
            texts_b.append(serialize_entity_with_name(hb, eb, max_hops))

    return names_a, names_b, hops_a_list, hops_b_list, texts_a, texts_b


def compute_metrics(ranks: np.ndarray) -> dict:
    return {
        "hit@1":  float(np.mean(ranks == 1)),
        "hit@5":  float(np.mean(ranks <= 5)),
        "hit@10": float(np.mean(ranks <= 10)),
        "mrr":    float(np.mean(1.0 / ranks)),
    }


def _ranks_for_alpha(reranked: list, in_top_k: np.ndarray, top_k: int, alpha: float) -> np.ndarray:
    ranks = []
    for i, query in enumerate(reranked):
        if not in_top_k[i]:
            ranks.append(top_k + 1)
            continue
        cands = sorted(
            query["candidates"],
            key=lambda c: (
                alpha * c["dpr_score"] + (1.0 - alpha) * c["llm_score"]
                if c.get("llm_score") is not None
                else c["dpr_score"]
            ),
            reverse=True,
        )
        rank = top_k + 1
        for r, cand in enumerate(cands):
            if cand["pool_idx"] == i:
                rank = r + 1
                break
        ranks.append(rank)
    return np.array(ranks)


@torch.no_grad()
def evaluate_hit(cfg: Config, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    alphas_str = getattr(args, "alphas", None)
    if alphas_str:
        alphas = [float(a.strip()) for a in alphas_str.split(",")]
    else:
        alphas = [cfg.hit_alpha]

    wandb.init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        name=f"DPR+LLM-Listwise-Hit@K-WithName-top{cfg.top_k}",
        config={
            "dpr_checkpoint": cfg.dpr_checkpoint_with_name,
            "top_k":          cfg.top_k,
            "max_hops":       cfg.max_hops,
            "alphas":         alphas,
            "use_name":       True,
        },
    )

    print(f"Loading model from: {cfg.dpr_checkpoint_with_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.dpr_checkpoint_with_name, use_fast=False)
    model = BiEncoderModel(cfg.dpr_checkpoint_with_name).to(device)
    model.eval()

    hit_eval_path = getattr(args, "hit_eval_path", None) or cfg.hit_eval_path
    print(f"Loading data from: {hit_eval_path}")
    names_a, names_b, hops_a_list, hops_b_list, texts_a, texts_b = load_hit_data(
        hit_eval_path, cfg.max_hops
    )
    N = len(names_a)
    print(f"Loaded {N} pairs.")

    print(f"\nEncoding {N} entity_b candidates (with name)...")
    emb_b = encode_corpus(texts_b, model, tokenizer, device, cfg.index_batch_size, cfg.max_length)

    print(f"Building FAISS index...")
    index = build_index(emb_b)

    print(f"Encoding {N} entity_a queries (with name)...")
    emb_a = encode_corpus(texts_a, model, tokenizer, device, cfg.index_batch_size, cfg.max_length)

    print(f"\nRetrieving top-{cfg.top_k} candidates per query...")
    top_indices, top_scores = retrieve_top_k(emb_a, index, cfg.top_k)

    in_top_k = np.array([i in top_indices[i] for i in range(N)])
    dpr_recall = float(np.mean(in_top_k))
    print(f"DPR recall@{cfg.top_k}: {dpr_recall * 100:.2f}% ({in_top_k.sum()}/{N} positives reachable)")

    queries = []
    for i in range(N):
        candidates = []
        for rank_0, (pool_idx, dpr_score) in enumerate(zip(top_indices[i], top_scores[i])):
            pool_idx = int(pool_idx)
            candidates.append({
                "entity_b":    names_b[pool_idx],
                "hops_b":      hops_b_list[pool_idx],
                "dpr_score":   float(dpr_score),
                "dpr_rank":    rank_0 + 1,
                "pool_idx":    pool_idx,
                "is_positive": pool_idx == i,
            })
        queries.append({
            "entity_a": names_a[i],
            "hops_a":   hops_a_list[i],
            "candidates": candidates,
        })

    print(f"\nReranking {N} queries × {cfg.top_k} candidates with LLM (with name, once for all alphas)...")
    reranked = rerank_queries(queries, cfg, use_name=True)

    all_metrics = {}
    for alpha in alphas:
        ranks = _ranks_for_alpha(reranked, in_top_k, cfg.top_k, alpha)
        m = compute_metrics(ranks)
        m["dpr_recall@k"] = dpr_recall
        all_metrics[alpha] = m

        print(f"\n===== α={alpha} — WithName =====")
        print(f"DPR Recall@{cfg.top_k} : {dpr_recall * 100:.2f}%")
        print(f"Hit@1  : {m['hit@1'] * 100:.2f}%")
        print(f"Hit@5  : {m['hit@5'] * 100:.2f}%")
        print(f"Hit@10 : {m['hit@10'] * 100:.2f}%")
        print(f"MRR    : {m['mrr']:.4f}")
        wandb.log({f"alpha_{alpha}/{k}": v for k, v in m.items()})

    wandb.finish()
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k",         type=int,   default=None)
    parser.add_argument("--alpha",          type=float, default=None,
                        help="단일 alpha 값 (기존 호환용)")
    parser.add_argument("--alphas",         type=str,   default=None,
                        help="콤마 구분 alpha 목록, LLM 1회 호출 후 모두 계산 (예: '1.0,0.75,0.5,0.25,0.0')")
    parser.add_argument("--checkpoint",     type=str,   default=None)
    parser.add_argument("--hit_eval_path",  type=str,   default=None,
                        help="Hit@K 평가 JSONL 경로 (기본값: cfg.hit_eval_path)")
    parser.add_argument("--llm_url",        type=str,   default=None,
                        help="LLM 서버 URL (기본값: cfg.llm_url)")
    parser.add_argument("--model",          type=str,   default=None,
                        help="LLM 모델 이름 (기본값: cfg.model)")
    args = parser.parse_args()

    cfg = Config()
    if args.top_k is not None:
        cfg.top_k = args.top_k
    if args.alpha is not None:
        cfg.hit_alpha = args.alpha
    if args.checkpoint is not None:
        cfg.dpr_checkpoint_with_name = args.checkpoint
    if args.llm_url is not None:
        cfg.llm_url = args.llm_url
    if args.model is not None:
        cfg.model = args.model

    evaluate_hit(cfg, args)
