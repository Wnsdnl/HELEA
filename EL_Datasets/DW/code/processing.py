import json
import re
from tqdm import tqdm

# Update path to match your local setup
INPUT_FILE = "./EL_Datasets/DW/DW_positive/english/DW_positive_en.jsonl"

if INPUT_FILE.endswith(".jsonl"):
    OUTPUT_FILE = INPUT_FILE.replace(".jsonl", "_cleaned.jsonl")
else:
    OUTPUT_FILE = INPUT_FILE + "_cleaned.jsonl"

QID_EXACT = re.compile(r"^Q\d+$")
PAREN_RE = re.compile(r"\s*\([^)]*\)")

def is_qid(text):
    """문자열이 정확히 QID 형태인지 확인"""
    if not isinstance(text, str): 
        return False
    return bool(QID_EXACT.match(text))

def has_qid_in_hop(hop_dict):
    """개별 1-hop 딕셔너리 안에 해결되지 않은 QID가 있는지 확인"""
    for val in hop_dict.values():
        if is_qid(val): 
            return True
    return False

def remove_parentheses(text, stats_counter):
    """괄호를 제거하고, 변경이 발생하면 카운터를 증가시키는 헬퍼 함수"""
    if not isinstance(text, str):
        return text
    cleaned_text = PAREN_RE.sub("", text).strip()
    
    if text != cleaned_text:
        stats_counter["parentheses_removed"] += 1
        
    return cleaned_text

def main():
    print("[시작] 전체 데이터 QID 정제 및 괄호 제거 작업")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}\n")
    
    total_lines = 0
    saved_lines = 0
    removed_lines = 0  
    removed_hops = 0   
    
    stats = {"parentheses_removed": 0}
    
    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
         
        # enumerate를 사용해 1번 줄부터 카운트 시작
        for i, line in enumerate(tqdm(fin, desc="Cleaning & Filtering"), start=1):
            line = line.strip()
            if not line: 
                continue
            
            total_lines += 1
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # [조건 1] 메인 타겟 엔티티 자체가 QID면 복구 불가 -> 데이터 통째로 삭제
            if is_qid(data.get("entity_a")) or is_qid(data.get("entity_b")):
                removed_lines += 1
                continue
            
            # 기존 1-hop 리스트 가져오기
            original_hops_a = data.get("1-hops_a", [])
            original_hops_b = data.get("1-hops_b", [])

            # [조건 2] 애초에 1-hop 리스트가 비어있는 놈이 있다면 데이터 삭제
            if not original_hops_a or not original_hops_b:
                removed_lines += 1
                continue
            
            # [조건 3] QID가 포함된 1-hop 정밀 제거
            cleaned_hops_a = [hop for hop in original_hops_a if not has_qid_in_hop(hop)]
            cleaned_hops_b = [hop for hop in original_hops_b if not has_qid_in_hop(hop)]
            
            # 솎아낸 1-hop 개수 누적 카운트
            removed_hops += (len(original_hops_a) - len(cleaned_hops_a))
            removed_hops += (len(original_hops_b) - len(cleaned_hops_b))
            
            # [조건 4] 불량 1-hop을 지우고 났더니 리스트가 비어버린 경우 -> 데이터 통째로 삭제
            if not cleaned_hops_a or not cleaned_hops_b:
                removed_lines += 1
                continue
            

            # [조건 5] 살아남은 데이터에서 괄호 정보 일괄 제거
            if "entity_a" in data:
                data["entity_a"] = remove_parentheses(data["entity_a"], stats)
            if "entity_b" in data:
                data["entity_b"] = remove_parentheses(data["entity_b"], stats)
                
            for hop in cleaned_hops_a:
                if "head" in hop:
                    hop["head"] = remove_parentheses(hop["head"], stats)
            for hop in cleaned_hops_b:
                if "head" in hop:
                    hop["head"] = remove_parentheses(hop["head"], stats)

            # 살아남은 데이터 업데이트 후 저장
            data["1-hops_a"] = cleaned_hops_a
            data["1-hops_b"] = cleaned_hops_b
            
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            saved_lines += 1

    print("\n=== 전체 데이터 정제 완료 ===")
    print(f"총 검사한 데이터 : {total_lines:,} 개")
    print("--------------------------------------------------")
    print(f"최종 살아남은 데이터 : {saved_lines:,} 개")
    print(f"통째로 삭제된 데이터 : {removed_lines:,} 개 (빈 리스트 또는 메인 엔티티 QID)")
    print(f"솎아낸 불량 1-hop  : {removed_hops:,} 개")
    print(f"괄호 정보가 제거된 총 횟수: {stats['parentheses_removed']:,} 번")
    print("--------------------------------------------------")
    print(f"결과 파일 저장됨   : {OUTPUT_FILE}")

if __name__ == "__main__":
    main()