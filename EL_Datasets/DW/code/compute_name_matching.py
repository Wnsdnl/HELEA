import json, re
from pathlib import Path

def normalize(name):
    return re.sub(r'\s+', ' ', name.strip().lower())

def predict_exact_then_sub(name_a, name_b):
    a, b = normalize(name_a), normalize(name_b)
    if a == b:
        return 1
    if a in b or b in a:
        return 1
    return 0

def evaluate(path):
    tp = fp = fn = tn = 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        pred = predict_exact_then_sub(d['entity_a'], d['entity_b'])
        label = d['label']
        if pred == 1 and label == 1: tp += 1
        elif pred == 1 and label == 0: fp += 1
        elif pred == 0 and label == 1: fn += 1
        else: tn += 1
    total = tp + fp + fn + tn
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}  total={total}")
    print(f"  Acc={acc*100:.2f}%  Prec={prec*100:.2f}%  Rec={rec*100:.2f}%  F1={f1*100:.2f}%")

print("=== DW-HN29K ===")
evaluate('EL_Datasets/DW/Benchmark/DW_1H_29k_EN.jsonl')

print("\n=== DY-HN27K ===")
evaluate('EL_Datasets/DY/DY_1H_27K_EN.jsonl')
