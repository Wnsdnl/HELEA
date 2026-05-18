import json
import random
import torch
from collections import defaultdict
from torch.utils.data import Dataset, Sampler


def serialize_entity(hops, entity_name, max_hops=None):
    hops = hops if isinstance(hops, list) else []
    formatted = []
    seen_relations = set()
    for h in hops:
        relation = h.get("relation", "")
        tail = h.get("tail", "")
        if str(tail) == str(entity_name):
            continue
        if relation in seen_relations:
            continue
        seen_relations.add(relation)
        formatted.append(f"{relation}: {tail}")
        if max_hops and len(formatted) >= max_hops:
            break
    return " | ".join(formatted) if formatted else ""


def serialize_entity_with_name(hops, entity_name, max_hops=None):
    triples = serialize_entity(hops, entity_name, max_hops)
    return f"{entity_name} | {triples}" if triples else entity_name


class EntityDataset(Dataset):
    def __init__(self, data_path, tokenizer, cfg):
        self.tokenizer = tokenizer
        self.max_length = cfg.max_length
        self.max_hops = cfg.max_hops
        self.use_name = getattr(cfg, 'use_name', False)

        print(f"Loading data from {data_path}...")
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = f.readlines()
        print(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def _tokenize(self, text):
        enc = self.tokenizer(
            text, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def __getitem__(self, idx):
        d = json.loads(self.data[idx])

        fn = serialize_entity_with_name if self.use_name else serialize_entity
        text_a = fn(d.get("1-hops_a", []), d.get("entity_a", ""), self.max_hops)
        text_b = fn(d.get("1-hops_b", []), d.get("entity_b", ""), self.max_hops)

        ids_a, mask_a = self._tokenize(text_a)
        ids_b, mask_b = self._tokenize(text_b)

        return {
            "ids_a":  ids_a,
            "mask_a": mask_a,
            "ids_b":  ids_b,
            "mask_b": mask_b,
            "label":  float(d.get("label", 0.0)),
        }


def collate_fn(batch):
    keys = ["ids_a", "mask_a", "ids_b", "mask_b"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    out["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return out


class PairWithHardNegSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.n = len(dataset.data)

        print("Building name index for hard negative sampling...")
        self.name_to_indices = defaultdict(list)
        self.index_to_name = {}
        for i, line in enumerate(dataset.data):
            name = json.loads(line)['entity_a']
            self.name_to_indices[name].append(i)
            self.index_to_name[i] = name
        print(f"Done. {len(self.name_to_indices):,} unique entity names indexed.")

    def __iter__(self):
        perm = torch.randperm(self.n).tolist()
        used = set()
        batch = []

        for idx in perm:
            if idx in used:
                continue

            batch.append(idx)
            used.add(idx)

            # [HARD-NEG] 같은 entity 이름의 샘플을 배치에 함께 넣는 로직
            # name = self.index_to_name[idx]
            # candidates = [i for i in self.name_to_indices[name] if i not in used]
            # if candidates:
            #     partner = random.choice(candidates)
            #     batch.append(partner)
            #     used.add(partner)

            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]

        if batch:
            yield batch

    def __len__(self):
        return self.n // self.batch_size
