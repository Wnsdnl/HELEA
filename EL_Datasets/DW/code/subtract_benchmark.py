import json
import hashlib
import os
from collections import Counter, defaultdict


def get_record_hash(data):
    """
    데이터의 핵심 필드(entity, 1-hops, label)를 추출하여 고유한 해시값을 생성합니다.
    """
    hops_a = sorted([json.dumps(h, sort_keys=True) for h in data.get("1-hops_a", [])])
    hops_b = sorted([json.dumps(h, sort_keys=True) for h in data.get("1-hops_b", [])])
    
    core_data = {
        "entity_a": data.get("entity_a"),
        "1-hops_a": hops_a,
        "entity_b": data.get("entity_b"),
        "1-hops_b": hops_b,
        "label": data.get("label")
    }
    
    canonical_str = json.dumps(core_data, sort_keys=True)
    return hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()


def remove_leaks_and_save(train_path, eval_path):
    # 학습 데이터 내에 똑같은 중복 데이터가 여러 번 등장할 수도 있으므로 리스트로 줄 번호 관리
    train_hashes = defaultdict(list)
    original_label_counts = Counter()
    
    print("1. 학습 데이터를 분석하여 고유 구조(Hash)를 저장하는 중...")
    with open(train_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            data = json.loads(line)
            label = data.get('label')
            
            record_hash = get_record_hash(data)
            train_hashes[record_hash].append(i)
            original_label_counts[label] += 1

    print(f"-> 총 {sum(original_label_counts.values()):,}개의 학습 데이터를 읽었습니다.")

    print("2. 검증 데이터와 구조적 완전 일치(Exact Match) 여부를 확인하는 중...")
    leaked_train_lines = set()
    total_eval = 0
    
    with open(eval_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_eval += 1
            data = json.loads(line)
            record_hash = get_record_hash(data)
            
            # 해시값이 존재하면, 해당 해시를 가진 모든 학습 데이터의 줄 번호를 누수 목록에 추가
            if record_hash in train_hashes:
                for line_num in train_hashes[record_hash]:
                    leaked_train_lines.add(line_num)

    print(f"-> 검증 데이터 {total_eval:,}개 중 일치 항목 발견. 제거해야 할 학습 데이터 라인은 총 {len(leaked_train_lines):,}개 입니다.\n")

    # 3. 새로운 파일명 생성 (_ver2 붙이기)
    base_name, ext = os.path.splitext(train_path)
    output_path = f"{base_name}_ver2{ext}"

    print(f"3. 중복 데이터를 제거하고 새로운 파일에 쓰는 중...\n   경로: {output_path}")
    clean_label_counts = Counter()
    saved_count = 0
    
    # 원본을 다시 읽으면서, 누수 목록에 없는 라인만 새 파일에 기록
    with open(train_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for i, line in enumerate(fin, 1):
            if i not in leaked_train_lines:
                fout.write(line)
                saved_count += 1
                
                # 라벨 분포 계산을 위해 파싱
                data = json.loads(line)
                clean_label_counts[data.get('label')] += 1

    # 4. 최종 결과 출력
    print("\n[필터링 완료 및 저장 결과]")
    print(f"- 원본 학습 데이터 수: {sum(original_label_counts.values()):,}개")
    print(f"- 제거된 중복 데이터 수: {len(leaked_train_lines):,}개")
    print(f"- 최종 저장된 데이터 수: {saved_count:,}개")
    
    if saved_count > 0:
        print("\n[최종 라벨 분포]")
        for label, count in clean_label_counts.items():
            ratio = (count / saved_count) * 100
            print(f"  * Label {label}: {count:,}개 ({ratio:.2f}%)")


if __name__ == "__main__":
    train_file = "./EL_Datasets/DW/DW_combined/english/DW_extended_en.jsonl"
    eval_file  = "./EL_Datasets/DW/Benchmark/DW_1H_30k_EN.jsonl"
    
    remove_leaks_and_save(train_file, eval_file)