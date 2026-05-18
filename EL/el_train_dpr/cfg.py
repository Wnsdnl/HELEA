import os

class Config:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    max_length = 256
    batch_size = 512
    learning_rate = 2e-5
    max_steps = 43000
    warmup_steps = 4300
    save_steps = 10750
    logging_steps = 25
    max_hops = 15
    scale = 20.0

    use_name = os.environ.get("USE_NAME", "0") == "1"

    # DW training data (name-hidden / name+triple controlled by USE_NAME env var)
    train_data_path = './EL_Datasets/DW/DW_combined/english/DW_extended_en.jsonl'
    save_dir = "./checkpoints/dw_name_hidden"
    save_dir_with_name = "./checkpoints/dw_name_triple"

    # DY training data
    # train_data_path = './EL_Datasets/DY/DY_extended_en.jsonl'
    # save_dir = "./checkpoints/dy_name_hidden"
    # save_dir_with_name = "./checkpoints/dy_name_triple"

    wandb_project = "entity-alignment"
    wandb_entity  = None  # set to your wandb username/team

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
