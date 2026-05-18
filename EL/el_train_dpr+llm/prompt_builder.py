"""
DPR+LLM 파이프라인용 listwise reranking 프롬프트.

format_entity_reranking: relation 중복 제거 + self-referential triple 제거.
build_listwise_prompt: 후보 K개를 한 프롬프트에 넣어 순위를 직접 출력.
parse_listwise_response: LLM 응답 텍스트 → 0-indexed 순위 리스트 변환 + 검증.
"""

import re

# ──────────────────────────────────────────────
# Pointwise scoring (eval_accuracy.py 용)
# ──────────────────────────────────────────────

RERANKING_SYSTEM_PROMPT = (
    "You are an expert in cross-knowledge-graph entity alignment. "
    "You will be given two entities from different knowledge graphs (DBpedia and Wikidata). "
    "Entity names are hidden — base your judgment EXCLUSIVELY on the knowledge graph triples. "
    "Entities with identical names can be completely different real-world objects. "
    "Output a single similarity score between 0.0 and 1.0 indicating how likely the two entities "
    "refer to the same real-world object. "
    "You MUST output exactly one line in this format: SCORE: <float between 0.0 and 1.0>"
)


def build_reranking_prompt(
    entity_a: str,
    hops_a: list,
    entity_b: str,
    hops_b: list,
    max_triples: int = 15,
) -> str:
    block_a = format_entity_reranking(entity_a, hops_a, max_triples, label="Entity A (name hidden)")
    block_b = format_entity_reranking(entity_b, hops_b, max_triples, label="Entity B (name hidden)")
    return (
        f"[Entity A]\n{block_a}\n\n"
        f"[Entity B]\n{block_b}\n\n"
        "How likely are these two entities the same real-world object? "
        "Output ONLY: SCORE: <float between 0.0 and 1.0>"
    )


RERANKING_SYSTEM_PROMPT_WITH_NAME = (
    "You are an expert in cross-knowledge-graph entity alignment. "
    "You will be given two entities from different knowledge graphs (DBpedia and Wikidata), "
    "including their names and knowledge graph triples. "
    "Output a single similarity score between 0.0 and 1.0 indicating how likely the two entities "
    "refer to the same real-world object. "
    "You MUST output exactly one line in this format: SCORE: <float between 0.0 and 1.0>"
)


def build_reranking_prompt_with_name(
    entity_a: str,
    hops_a: list,
    entity_b: str,
    hops_b: list,
    max_triples: int = 15,
) -> str:
    block_a = format_entity_reranking(entity_a, hops_a, max_triples)
    block_b = format_entity_reranking(entity_b, hops_b, max_triples)
    return (
        f"[Entity A]\n{block_a}\n\n"
        f"[Entity B]\n{block_b}\n\n"
        "How likely are these two entities the same real-world object? "
        "Output ONLY: SCORE: <float between 0.0 and 1.0>"
    )


def parse_response(response: str) -> dict:
    """Parse 'SCORE: 0.85' from LLM response. Returns {"score": float} in [0, 1]."""
    match = re.search(r"score\s*:\s*([0-9]*\.?[0-9]+)", response, re.IGNORECASE)
    if match:
        score = float(match.group(1))
        score = max(0.0, min(1.0, score))
    else:
        score = 0.5  # 파싱 실패 시 중간값으로 fallback
    return {"score": score}


# ──────────────────────────────────────────────
# Listwise reranking (eval_hit.py / reranker.py 용)
# ──────────────────────────────────────────────

LISTWISE_SYSTEM_PROMPT = (
    "You are an expert in cross-knowledge-graph entity alignment. "
    "A query entity from DBpedia and several candidate entities from Wikidata are given. "
    "Entity names are hidden — base your judgment EXCLUSIVELY on the knowledge graph triples. "
    "Entities with identical names can be completely different real-world objects "
    "(e.g., two different people named the same, two cities with the same name). "
    "The DPR scores are a rough similarity hint — your primary task is to rank by "
    "how well each candidate's triples match the query entity's triples. "
    "You MUST begin your response immediately with 'RANKING:' — do NOT generate any analysis, "
    "explanation, or reasoning before the RANKING line. "
    "If you reason first, your final line MUST be 'RANKING:' (uppercase, exact) followed by "
    "comma-separated <number>:<score> pairs only. "
    "Do not copy or repeat triples from the prompt. "
    "For each candidate, write at most two sentences of analysis."
)


def format_entity_reranking(name: str, hops: list, max_triples: int = 15, label: str = None) -> str:
    display = label if label is not None else name
    lines = [f"Name: {display}", "Triples:"]
    seen_relations = set()
    count = 0
    for hop in hops:
        relation = hop.get("relation", "")
        tail = hop.get("tail", "")
        if str(tail) == str(name):
            continue
        if relation in seen_relations:
            continue
        seen_relations.add(relation)
        lines.append(f"  - {relation}: {tail}")
        count += 1
        if count >= max_triples:
            break
    if not count:
        lines.append("  (no triples available)")
    return "\n".join(lines)


def build_listwise_prompt(
    entity_a: str,
    hops_a: list,
    candidates: list,
    max_triples: int = 15,
) -> str:
    """
    Build a listwise reranking prompt: one call with all K candidates.
    Expects candidates sorted by DPR rank (dpr_rank field, 1-indexed).
    """
    query_block = format_entity_reranking(entity_a, hops_a, max_triples, label="Query Entity (name hidden)")
    cand_blocks = []
    for c in candidates:
        block = format_entity_reranking(c["entity_b"], c["hops_b"], max_triples, label=f"Candidate {c['dpr_rank']} (name hidden)")
        cand_blocks.append(
            f"[Candidate {c['dpr_rank']}] DPR cosine sim: {c['dpr_score']:.3f}\n{block}"
        )

    k = len(candidates)
    return (
        f"[Query Entity]\n{query_block}\n\n"
        + "\n\n".join(cand_blocks)
        + f"\n\nRank the {k} candidates from most to least likely to be the same "
        "real-world entity as the Query Entity. "
        "Base your ranking primarily on how well each candidate's triples match the query. "
        "DPR scores are a secondary hint only.\n\n"
        "Then output RANKING.\n\n"
        "Output ONLY the following format — no other text before or after:\n"
        "RANKING: <number>:<score>, <number>:<score>, ... (best match first; score is 0.0–1.0 similarity — do NOT default to sequential order)\n"
        "ENTITY_TYPE: <person / place / organization / event / other>\n"
        "REASONING: <2-3 sentences: identify the 2-3 most discriminating triples from the query, "
        "then explain which candidate best matches and why others do not>\n"
    )


LISTWISE_SYSTEM_PROMPT_WITH_NAME = (
    "You are an expert in cross-knowledge-graph entity alignment. "
    "A query entity from DBpedia and several candidate entities from Wikidata are given, "
    "including their names and knowledge graph triples. "
    "The DPR scores are a rough similarity hint — your primary task is to rank by "
    "how well each candidate's triples and name match the query entity. "
    "You MUST begin your response immediately with 'RANKING:' — do NOT generate any analysis, "
    "explanation, or reasoning before the RANKING line. "
    "If you reason first, your final line MUST be 'RANKING:' (uppercase, exact) followed by "
    "comma-separated <number>:<score> pairs only. "
    "Do not copy or repeat triples from the prompt. "
    "For each candidate, write at most two sentences of analysis."
)


def build_listwise_prompt_with_name(
    entity_a: str,
    hops_a: list,
    candidates: list,
    max_triples: int = 15,
) -> str:
    query_block = format_entity_reranking(entity_a, hops_a, max_triples)
    cand_blocks = []
    for c in candidates:
        block = format_entity_reranking(c["entity_b"], c["hops_b"], max_triples)
        cand_blocks.append(
            f"[Candidate {c['dpr_rank']}] DPR cosine sim: {c['dpr_score']:.3f}\n{block}"
        )

    k = len(candidates)
    return (
        f"[Query Entity]\n{query_block}\n\n"
        + "\n\n".join(cand_blocks)
        + f"\n\nRank the {k} candidates from most to least likely to be the same "
        "real-world entity as the Query Entity. "
        "Base your ranking on how well each candidate's name and triples match the query. "
        "DPR scores are a secondary hint only.\n\n"
        "Then output RANKING.\n\n"
        "Output ONLY the following format — no other text before or after:\n"
        "RANKING: <number>:<score>, <number>:<score>, ... (best match first; score is 0.0–1.0 similarity — do NOT default to sequential order)\n"
        "ENTITY_TYPE: <person / place / organization / event / other>\n"
        "REASONING: <2-3 sentences: identify the 2-3 most discriminating triples from the query, "
        "then explain which candidate best matches and why others do not>\n"
    )


def parse_listwise_response(response: str, k: int) -> tuple[list, dict] | tuple[None, None]:
    """
    Parse 'RANKING: 3:0.92,1:0.75,...' from LLM response.
    Accepts partial rankings (token cutoff): fills missing positions with DPR order.
    Returns (ranking, scores) where:
      ranking: 0-indexed list of length k
      scores:  dict mapping 0-indexed candidate idx -> llm_score (only parsed candidates)
    Returns (None, None) if no valid entries found.
    """
    match = re.search(r"\branking:\s*([\d\.,:\s]+)", response, re.IGNORECASE)
    if not match:
        return None, None
    try:
        seen = set()
        ranking = []
        scores = {}
        for token in match.group(1).split(","):
            token = token.strip()
            if not token:
                continue
            if ":" in token:
                parts = token.split(":", 1)
                num_str = parts[0].strip()
                score_str = parts[1].strip()
                if not num_str.isdigit():
                    continue
                n = int(num_str) - 1
                if 0 <= n < k and n not in seen:
                    seen.add(n)
                    ranking.append(n)
                    try:
                        scores[n] = max(0.0, min(1.0, float(score_str)))
                    except ValueError:
                        pass
            else:
                if not token.isdigit():
                    continue
                n = int(token) - 1
                if 0 <= n < k and n not in seen:
                    seen.add(n)
                    ranking.append(n)
        if not ranking:
            return None, None
        remaining = [i for i in range(k) if i not in seen]
        return ranking + remaining, scores
    except (ValueError, AttributeError):
        return None, None
