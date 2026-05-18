import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import autocast

from cfg import Config
from model import BiEncoderModel
from dataset import EntityDataset, collate_fn, PairWithHardNegSampler
from utils import save_model
import wandb


def info_nce_loss(logits, pos_indices):
    loss_a2b = F.cross_entropy(logits[pos_indices], pos_indices)
    loss_b2a = F.cross_entropy(logits.T[pos_indices], pos_indices)
    return (loss_a2b + loss_b2a) / 2.0, loss_a2b, loss_b2a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-name", action="store_true", help="Include entity name in serialized text")
    args = parser.parse_args()

    cfg = Config()
    if args.use_name:
        cfg.use_name = True
        cfg.save_dir = cfg.save_dir_with_name

    os.makedirs(cfg.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_name = "Dpr-MiniLM-WithName-experiments" if cfg.use_name else "Dpr-MiniLM-Serialize-experiments(random hard Negative)"
    wandb.init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        name=run_name,
        config={
            "model_name": cfg.model_name,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "max_steps": cfg.max_steps,
            "warmup_steps": cfg.warmup_steps,
            "init_scale": cfg.scale,
            "max_hops": cfg.max_hops,
            "train_data_path": cfg.train_data_path,
            "use_name": cfg.use_name,
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = BiEncoderModel(cfg.model_name, init_scale=cfg.scale)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    base_model = model.module if hasattr(model, 'module') else model

    dataset = EntityDataset(cfg.train_data_path, tokenizer, cfg)
    sampler = PairWithHardNegSampler(dataset, cfg.batch_size)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=True
    )

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.max_steps
    )

    model.train()
    print(f"Start training\n- InfoNCE Loss with Learnable Temperature\n- Serialize Hops\n- Batch Size: {cfg.batch_size}")

    step = 0
    while step < cfg.max_steps:
      for batch in loader:
        if step >= cfg.max_steps:
            break

        ids_a  = batch["ids_a"].to(device)
        mask_a = batch["mask_a"].to(device)
        ids_b  = batch["ids_b"].to(device)
        mask_b = batch["mask_b"].to(device)
        labels = batch["labels"].to(device)

        B = ids_a.shape[0]
        pos_indices = (labels == 1).nonzero(as_tuple=True)[0]

        if len(pos_indices) == 0:
            step += 1
            continue

        combined_ids  = torch.cat([ids_a, ids_b], dim=0)   # (2B, L)
        combined_mask = torch.cat([mask_a, mask_b], dim=0)

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            combined_emb = model(combined_ids, combined_mask)  # (2B, D)
            emb_a = combined_emb[:B]   # (B, D)
            emb_b = combined_emb[B:]   # (B, D)

            sim_matrix = torch.matmul(emb_a, emb_b.T)         # (B, B)
            logits = sim_matrix * base_model.log_scale.exp()
            loss, loss_a2b, loss_b2a = info_nce_loss(logits, pos_indices)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        wandb.log({
            "loss/total": loss.item(),
            "loss/a2b": loss_a2b.item(),
            "loss/b2a": loss_b2a.item(),
            "temperature": (1.0 / base_model.log_scale.exp()).item(),
            "scale": base_model.log_scale.exp().item()
        }, step=step)

        if step % cfg.logging_steps == 0:
            curr_lr = scheduler.get_last_lr()[0]
            scale = base_model.log_scale.exp().item()
            print(f"Step [{step}/{cfg.max_steps}] | Loss: {loss.item():.4f} (a2b: {loss_a2b.item():.4f}, b2a: {loss_b2a.item():.4f}) | Scale: {scale:.3f} | LR: {curr_lr:.8f}")

        if step > 0 and step % cfg.save_steps == 0:
            print("Saving intermediate model...")
            save_model(model, tokenizer, cfg.save_dir, step)

        step += 1

    print("Training finished. Saving final model...")
    save_model(model, tokenizer, cfg.save_dir, "final")

    wandb.finish()


if __name__ == "__main__":
    main()
