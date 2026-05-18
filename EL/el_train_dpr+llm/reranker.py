"""
DPR 후보에 대한 LLM listwise 재정렬.

후보 K개를 한 프롬프트에 넣어 순위를 직접 출력받으므로
LLM 호출 수가 N×K → N으로 감소한다.
LLM 응답 파싱 실패 시 DPR 순서로 fallback.
"""

import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "Utils/llm"))
from core.async_vllm_client import AsyncVLLMClient

from prompt_builder import (
    build_listwise_prompt, build_listwise_prompt_with_name,
    parse_listwise_response,
    LISTWISE_SYSTEM_PROMPT, LISTWISE_SYSTEM_PROMPT_WITH_NAME,
)


def fuse_scores(dpr_score: float, llm_score: float, alpha: float) -> float:
    """alpha * dpr_score + (1 - alpha) * llm_score"""
    return alpha * dpr_score + (1.0 - alpha) * llm_score


def rerank_queries_listwise(queries: list, cfg, use_name: bool = False) -> list:
    client = AsyncVLLMClient(cfg)

    prompt_fn     = build_listwise_prompt_with_name if use_name else build_listwise_prompt
    system_prompt = LISTWISE_SYSTEM_PROMPT_WITH_NAME if use_name else LISTWISE_SYSTEM_PROMPT

    all_prompts = [
        prompt_fn(
            entity_a=q["entity_a"],
            hops_a=q["hops_a"],
            candidates=q["candidates"],
            max_triples=cfg.max_triples,
        )
        for q in queries
    ]

    print(f"Sending {len(all_prompts)} LLM requests (listwise, concurrency={cfg.concurrency})...")
    raw_responses = client.run(
        all_prompts,
        system_prompt=system_prompt,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )

    results = [dict(q, candidates=[dict(c) for c in q["candidates"]]) for q in queries]

    print("=== SAMPLE LLM RESPONSE (첫 번째) ===")
    print(raw_responses[0])
    print("=====================================")

    parse_failures = 0
    partial_scores = 0
    top1_changed = 0
    failure_samples_printed = 0
    for q_idx, response in enumerate(raw_responses):
        candidates = results[q_idx]["candidates"]
        k = len(candidates)
        ranking, llm_scores = parse_listwise_response(response, k)

        if ranking is None:
            parse_failures += 1
            if failure_samples_printed < 2:
                print(f"=== PARSE 실패 샘플 {failure_samples_printed + 1} ===")
                print(response)
                print("===============================")
                failure_samples_printed += 1
            continue

        if len(llm_scores) < k:
            partial_scores += 1

        # 각 후보에 fusion score 계산:
        # LLM 점수가 있으면 fusion, 없으면 DPR score만 사용 (partial fallback)
        for i, cand in enumerate(candidates):
            if i in llm_scores:
                cand["llm_score"]    = llm_scores[i]
                cand["fusion_score"] = fuse_scores(cand["dpr_score"], llm_scores[i], cfg.hit_alpha)
            else:
                cand["llm_score"]    = None
                cand["fusion_score"] = cand["dpr_score"]

        results[q_idx]["candidates"] = sorted(candidates, key=lambda c: c["fusion_score"], reverse=True)

        if results[q_idx]["candidates"][0]["dpr_rank"] != 1:
            top1_changed += 1

    n = len(queries)
    if parse_failures:
        print(f"Warning: {parse_failures}/{n} queries fell back to DPR order (parse failure)")
    if partial_scores:
        print(f"Warning: {partial_scores}/{n} queries had partial LLM scores (truncation)")
    print(f"LLM changed DPR top-1: {top1_changed}/{n} ({top1_changed / n * 100:.1f}%)")

    return results
