import os
import torch
import torch.nn.functional as F
from torch.amp import autocast
from dataset import serialize_entity


def save_model(model, tokenizer, save_dir, suffix):
    ckpt_path = os.path.join(save_dir, f"checkpoint-{suffix}")
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)


@torch.no_grad()
def encode_hops_batch(raw_hops_list, entity_names, model, tokenizer, device, cfg):
    """
    Serialize each entity's hops into one string, encode with the model.
    Returns (N_entities, D) L2-normalized embeddings (float32, CPU).
    """
    all_embs = []
    batch_size = cfg.batch_size * 2
    L = cfg.max_length
    use_cuda = torch.cuda.is_available()
    ctx = autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda)

    for i in range(0, len(raw_hops_list), batch_size):
        batch_hops  = raw_hops_list[i:i + batch_size]
        batch_names = entity_names[i:i + batch_size]

        texts = [
            serialize_entity(hops, name, cfg.max_hops)
            for hops, name in zip(batch_hops, batch_names)
        ]

        enc = tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=L, return_tensors="pt"
        )
        ids  = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        with ctx:
            emb = model(ids, mask)   # (B, D)

        all_embs.append(emb.float().cpu())

        if i > 0 and i % (batch_size * 20) == 0:
            print(f"  Encoded {i}/{len(raw_hops_list)}")

    return torch.cat(all_embs, dim=0)   # (N_entities, D)
