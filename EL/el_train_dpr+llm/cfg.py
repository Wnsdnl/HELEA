import os

_HERE = os.path.dirname(os.path.abspath(__file__))  # el_train_dpr+llm/
_EL   = os.path.dirname(_HERE)                       # EL/
_ROOT = os.path.dirname(_EL)                         # repo root

class Config:
    # --- DPR inference ---
    max_length       = 256
    max_hops         = 15

    # DW checkpoints (set by train.py; name-hidden vs name+triple via USE_NAME)
    dpr_checkpoint           = os.path.join(_EL, 'el_train_dpr/checkpoints/dw_name_hidden/checkpoint-final')
    dpr_checkpoint_with_name = os.path.join(_EL, 'el_train_dpr/checkpoints/dw_name_triple/checkpoint-final')

    # DY checkpoints
    # dpr_checkpoint           = os.path.join(_EL, 'el_train_dpr/checkpoints/dy_name_hidden/checkpoint-final')
    # dpr_checkpoint_with_name = os.path.join(_EL, 'el_train_dpr/checkpoints/dy_name_triple/checkpoint-final')

    index_batch_size = 256

    # --- Reranking ---
    top_k     = 10    # number of DPR candidates passed to LLM
    hit_alpha = 0.75  # DPR weight for Hit@K evaluation (best per Table 4 ablation)
    acc_alpha = 0.75  # DPR weight for Accuracy/F1 evaluation

    # --- LLM (AsyncVLLMClient reads: llm_url, model, concurrency) ---
    llm_url     = "http://localhost:8005"  # update to your vLLM endpoint
    model       = "google/gemma-4-31B-it"
    concurrency = 32
    max_triples = 15
    temperature = 0.0
    max_tokens  = 1024

    # --- Evaluation data ---
    # DW benchmarks
    hit_eval_path = os.path.join(_ROOT, 'EL_Datasets/DW/Benchmark/DW_1H_15k_EN.jsonl')
    acc_eval_path = os.path.join(_ROOT, 'EL_Datasets/DW/Benchmark/DW_1H_29k_EN.jsonl')

    # DY benchmarks
    # hit_eval_path = os.path.join(_ROOT, 'EL_Datasets/DY/DY_1H_15K_EN.jsonl')
    # acc_eval_path = os.path.join(_ROOT, 'EL_Datasets/DY/DY_1H_27K_EN.jsonl')

    # --- WandB ---
    wandb_project = "entity-alignment"
    wandb_entity  = None  # set to your wandb username/team

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
