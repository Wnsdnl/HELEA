"""
FAISS 기반 코퍼스 인코딩 및 top-K 검색.

IndexFlatIP (내적 검색) 사용.
BiEncoderModel이 L2 정규화된 벡터를 출력하므로 내적 = cosine 유사도.

faiss 미설치 시 numpy matmul로 fallback.
"""

import numpy as np
import torch
from torch.amp import autocast

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    print("[indexer] faiss not found — using numpy fallback for retrieval.")


@torch.no_grad()
def encode_corpus(
    texts: list,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = 256,
    max_length: int = 256,
) -> np.ndarray:
    """Encode list of texts into L2-normalized float32 numpy array (N, D)."""
    all_embs = []
    use_cuda = torch.cuda.is_available()
    ctx = autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda)

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tok = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ids  = tok["input_ids"].to(device)
        mask = tok["attention_mask"].to(device)

        with ctx:
            emb = model(ids, mask)

        all_embs.append(emb.cpu().float())

        if i > 0 and i % (batch_size * 20) == 0:
            print(f"  Encoded {i}/{len(texts)}")

    return torch.cat(all_embs, dim=0).numpy()  # (N, D) float32


def build_index(embeddings: np.ndarray):
    """
    Build a search index from (N, D) float32 embeddings.
    Returns a faiss.IndexFlatIP if available, otherwise the embeddings array itself.
    """
    if _FAISS_AVAILABLE:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index
    else:
        return embeddings  # numpy fallback: store raw embeddings


def retrieve_top_k(
    query_embs: np.ndarray,
    index,
    k: int,
) -> tuple:
    """
    Retrieve top-K corpus entries per query.

    Returns:
        indices: (N_queries, k) int64
        scores:  (N_queries, k) float32  — cosine similarities in [-1, 1]
    """
    if _FAISS_AVAILABLE:
        scores, indices = index.search(query_embs, k)
        return indices, scores
    else:
        # numpy fallback: index is the raw (N_corpus, D) embeddings array
        sim = query_embs @ index.T  # (N_queries, N_corpus)
        top_idx = np.argsort(-sim, axis=1)[:, :k]
        top_scores = np.take_along_axis(sim, top_idx, axis=1)
        return top_idx.astype(np.int64), top_scores.astype(np.float32)
