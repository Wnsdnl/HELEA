# HELEA: Hard-Negative Benchmarks and LLM-Enhanced Entity Alignment

Code for our paper: **"HELEA: Hard-Negative Benchmarks and LLM-Enhanced Entity Alignment"**

Our Benchmark is available at https://huggingface.co/datasets/anonymous-submission2026/helea-benchmark

## Overview

HELEA is a two-stage entity alignment framework:
1. **HELEA-Retriever**: BiEncoder (all-MiniLM-L6-v2) trained with bidirectional InfoNCE loss on DW-Extended (5.57M pairs)
2. **HELEA-Reranker**: Listwise LLM reranking (Gemma 4 31B Instruct via vLLM) with score fusion

We also introduce two hard-negative benchmarks: **DW-HN29K** and **DY-HN27K**, where entity names are shared between positive and negative pairs, requiring models to reason from KG structure rather than surface names.

## Requirements

```bash
pip install -r requirements.txt
```

## Directory Structure

```
EL/
  el_train_dpr/          # BiEncoder training
  el_train_dpr+llm/      # DPR + LLM reranking & evaluation
EL_Datasets/DW/code/     # Dataset construction pipeline
Utils/llm/core/          # Async vLLM client
```

## Training (DPR Retriever)

```bash
# Name-hidden setting (USE_NAME=0, default)
cd EL/el_train_dpr && python train.py

# Name+triple setting
USE_NAME=1 python train.py
```

Update `cfg.py` to set `train_data_path` and `save_dir` for your environment.

## Evaluation

**Hit@K on DW-15K:**
```bash
python EL/el_train_dpr+llm/eval_hit.py
python EL/el_train_dpr+llm/eval_hit_with_name.py  # name+triple
```

**Accuracy/F1 on DW-HN29K:**
```bash
python EL/el_train_dpr+llm/eval_accuracy.py
python EL/el_train_dpr+llm/eval_accuracy_with_name.py  # name+triple
```

Update `EL/el_train_dpr+llm/cfg.py` for checkpoint paths and evaluation data paths. DY benchmark paths are provided as commented-out alternatives.

## LLM Serving

We serve Gemma 4 31B Instruct via vLLM:

```bash
vllm serve google/gemma-4-31B-it --port 8005
```

Update `cfg.py` (`llm_url`, `model`) to match your endpoint.

## Dataset Construction

Scripts in `EL_Datasets/DW/code/` reproduce the DW-Extended training corpus and DW-HN29K benchmark:

```
make_1hop.py            → extract 1-hop triples from KG dumps
processing.py           → clean & filter
combine_and_shuffle.py  → merge positive + negative pairs
subtract_benchmark.py   → remove benchmark overlap from training data
```

## Datasets

Benchmark files are released on Huggingface(https://huggingface.co/datasets/anonymous-submission2026/helea-benchmark).

## Results

| Model | DW-15K Hit@1 | DW-HN29K F1 | DY-15K Hit@1 | DY-HN27K F1 |
|-------|-------------|-------------|-------------|-------------|
| HELEA-Retriever | 0.987 | 0.853 | 0.992 | 0.719 |
| HELEA | 0.993 | 0.967 | 0.992 | 0.931 |
