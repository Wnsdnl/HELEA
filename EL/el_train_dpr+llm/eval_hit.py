"""
Hit@1 / Hit@5 평가: DPR 검색 + LLM listwise 재정렬.

Pipeline:
  1. DPR 모델 로드 (cfg.dpr_checkpoint)
  2. entity_b 15K개 인코딩 → FAISS 인덱스 구축
  3. entity_a 15K개 인코딩 → query 벡터
  4. 각 query i에 대해 FAISS top-K 검색 → 후보 인덱스 + DPR 점수
  5. LLM이 후보 K개를 listwise 재정렬 (reranker.rerank_queries_listwise)
  6. 재정렬된 리스트에서 정답(entity_b[i])의 순위 측정
  7. DPR Recall@K, Hit@1, Hit@5 출력

정답이 top-K 밖에 있는 경우:
  LLM 재정렬로 복구 불가 → rank = K+1 로 처리 (보수적 하한)
  DPR Recall@K = LLM이 재정렬 가능한 정답의 비율
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
from dataset import serialize_entity, serialize_entity_with_name

from cfg import Config
from indexer import encode_corpus, build_index, retrieve_top_k
from reranker import rerank_queries_listwise as rerank_queries


def load_hit_data(path: str, max_hops: int, use_name: bool = False):
    """
    Load DW_1H_15k_EN.jsonl.
    Returns parallel lists: entity names, hops (for LLM prompts), serialized texts (for DPR).
    All lists are aligned: index i corresponds to the i-th query/candidate pair.
    """
    _ser = serialize_entity_with_name if use_name else serialize_entity
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
            texts_a.append(_ser(ha, ea, max_hops))
            texts_b.append(_ser(hb, eb, max_hops))

    return names_a, names_b, hops_a_list, hops_b_list, texts_a, texts_b


def compute_metrics(ranks: np.ndarray) -> dict:
    return {
        "hit@1":  float(np.mean(ranks == 1)),
        "hit@5":  float(np.mean(ranks <= 5)),
        "hit@10": float(np.mean(ranks <= 10)),
        "mrr":    float(np.mean(1.0 / ranks)),
    }


def _ranks_for_alpha(reranked: list, in_top_k: np.ndarray, top_k: int, alpha: float) -> np.ndarray:
    """Recompute ranks using a given alpha without re-calling LLM."""
    ranks = []
    for i, query in enumerate(reranked):
        if not in_top_k[i]:
            ranks.append(top_k + 1)
            continue
        # Re-sort candidates using this alpha.
        # llm_score may be None if LLM parse failed; fall back to DPR-only for that candidate.
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

    # Support --alphas "1.0,0.75,0.5,0.25,0.0" for single-LLM-call multi-alpha sweep.
    alphas_str = getattr(args, "alphas", None)
    if alphas_str:
        alphas = [float(a.strip()) for a in alphas_str.split(",")]
    else:
        alphas = [cfg.hit_alpha]

    wandb.init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        name=f"DPR+LLM-Listwise-Hit@K-top{cfg.top_k}",
        config={
            "dpr_checkpoint": cfg.dpr_checkpoint,
            "top_k":          cfg.top_k,
            "max_hops":       cfg.max_hops,
            "alphas":         alphas,
        },
    )

    print(f"Loading model from: {cfg.dpr_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.dpr_checkpoint, use_fast=False)
    model = BiEncoderModel(cfg.dpr_checkpoint).to(device)
    model.eval()

    print(f"Loading data from: {cfg.hit_eval_path}")
    use_name = getattr(args, "show_names", False)
    _ser = serialize_entity_with_name if use_name else serialize_entity

    names_a, names_b, hops_a_list, hops_b_list, texts_a, texts_b = load_hit_data(
        cfg.hit_eval_path, cfg.max_hops, use_name=use_name
    )
    N = len(names_a)
    print(f"Loaded {N} pairs.")

    # Augmented pool: add unique hard-neg entity_b not already in positive pool
    if getattr(args, "hn_path", None):
        eb_set = set(names_b)
        with open(args.hn_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("label", 1) == 0 and d["entity_b"] not in eb_set:
                    eb = d["entity_b"]
                    hb = d.get("1-hops_b", [])
                    names_b.append(eb)
                    hops_b_list.append(hb)
                    texts_b.append(_ser(hb, eb, cfg.max_hops))
                    eb_set.add(eb)
        print(f"Augmented pool: +{len(names_b) - N} hard-neg entity_b (pool size: {len(names_b)})")

    emb_b = encode_corpus(texts_b, model, tokenizer, device, cfg.index_batch_size, cfg.max_length)

    print(f"Building FAISS index...")
    index = build_index(emb_b)

    print(f"Encoding {N} entity_a queries...")
    emb_a = encode_corpus(texts_a, model, tokenizer, device, cfg.index_batch_size, cfg.max_length)

    print(f"\nRetrieving top-{cfg.top_k} candidates per query...")
    top_indices, top_scores = retrieve_top_k(emb_a, index, cfg.top_k)

    in_top_k = np.array([i in top_indices[i] for i in range(N)])
    dpr_recall = float(np.mean(in_top_k))
    print(f"DPR recall@{cfg.top_k}: {dpr_recall * 100:.2f}% ({in_top_k.sum()}/{N} positives reachable)")

    # Build query dicts for reranker
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

    use_name = getattr(args, "show_names", False)
    print(f"\nReranking {N} queries × {cfg.top_k} candidates with LLM (use_name={use_name})...")
    # LLM is called ONCE; reranker now stores llm_score per candidate for alpha re-use.
    reranked = rerank_queries(queries, cfg, use_name=use_name)

    # Sweep all alphas without re-calling LLM
    all_metrics = {}
    for alpha in alphas:
        ranks = _ranks_for_alpha(reranked, in_top_k, cfg.top_k, alpha)
        m = compute_metrics(ranks)
        m["dpr_recall@k"] = dpr_recall
        all_metrics[alpha] = m

        print(f"\n===== α={alpha} =====")
        print(f"DPR Recall@{cfg.top_k} : {dpr_recall * 100:.2f}%")
        print(f"Hit@1  : {m['hit@1'] * 100:.2f}%")
        print(f"Hit@5  : {m['hit@5'] * 100:.2f}%")
        print(f"Hit@10 : {m['hit@10'] * 100:.2f}%" if 'hit@10' in m else "")
        print(f"MRR    : {m['mrr']:.4f}")
        wandb.log({f"alpha_{alpha}/{k}": v for k, v in m.items()})

    print("\nHit@1/Hit@5/Hit@10/MRR evaluation successfully logged to WandB!")
    wandb.finish()
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k",    type=int,   default=None)
    parser.add_argument("--alpha",    type=float, default=None,
                        help="단일 alpha 값 (기존 호환용)")
    parser.add_argument("--alphas",   type=str,   default=None,
                        help="콤마 구분 alpha 목록, LLM 1회 호출 후 모두 계산 (예: '1.0,0.75,0.5,0.25,0.0')")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--show-names", action="store_true",
                        help="LLM 프롬프트에 entity name 노출 (E4 ablation용)")
    parser.add_argument("--hn_path", type=str, default=None,
                        help="Hard-negative JSONL 경로 (augmented pool 평가용)")
    parser.add_argument("--hit_eval_path", type=str, default=None,
                        help="Hit@K 평가 JSONL 경로 (기본값: cfg.hit_eval_path)")
    args = parser.parse_args()

    cfg = Config()
    if args.top_k is not None:
        cfg.top_k = args.top_k
    if args.alpha is not None:
        cfg.hit_alpha = args.alpha
    if args.checkpoint is not None:
        cfg.dpr_checkpoint = args.checkpoint
    if args.hit_eval_path is not None:
        cfg.hit_eval_path = args.hit_eval_path

    evaluate_hit(cfg, args)
