import os
import random
from tqdm import tqdm

# Update paths to match your local setup
POS_FILE  = "./EL_Datasets/DW/DW_positive/english/DW_positive_en.jsonl"

NEG_FILE_1 = "./EL_Datasets/DW/DW_negative/english/DW_negative_en.jsonl"
NEG_FILE_2 = "./EL_Datasets/DW/DW_negative/english/DW_negative_extended_en.jsonl"

OUT_DIR   = "./EL_Datasets/DW/DW_combined/english"
OUT_FILE_1 = f"{OUT_DIR}/DW_en.jsonl"
OUT_FILE_2 = f"{OUT_DIR}/DW_extended_en.jsonl"

# 출력 디렉토리가 없으면 생성
os.makedirs(OUT_DIR, exist_ok=True)

def combine_and_shuffle(pos_path, neg_path, out_path):
    print(f"작업을 시작합니다: {os.path.basename(out_path)} 생성 중...")
    
    combined_lines = []
    
    # 1. 긍정(Positive) 데이터 읽기
    print(f"긍정 데이터 읽는 중: {pos_path}")
    with open(pos_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Positive"):
            line = line.strip()
            if line:
                combined_lines.append(line)
                
    # 2. 부정(Negative) 데이터 읽기
    print(f"부정 데이터 읽는 중: {neg_path}")
    with open(neg_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Negative"):
            line = line.strip()
            if line:
                combined_lines.append(line)
                
    # 3. 셔플 (랜덤 섞기)
    print(f"총 {len(combined_lines):,}개의 데이터를 섞고 있습니다...")
    random.shuffle(combined_lines)
    
    # 4. 결과 저장
    print(f"섞인 데이터를 저장하는 중: {out_path}")
    with open(out_path, 'w', encoding='utf-8') as f:
        for line in tqdm(combined_lines, desc="Saving"):
            f.write(line + '\n')
            
    print(f"완료! 저장 위치: {out_path}\n")

def main():
    # 첫 번째 조합: POS + NEG 1 -> DW.jsonl
    combine_and_shuffle(POS_FILE, NEG_FILE_1, OUT_FILE_1)
    
    # 두 번째 조합: POS + NEG 2 -> DW_extended.jsonl
    combine_and_shuffle(POS_FILE, NEG_FILE_2, OUT_FILE_2)
    
    print("모든 병합 및 셔플 작업이 끝났습니다!")

if __name__ == "__main__":
    main()