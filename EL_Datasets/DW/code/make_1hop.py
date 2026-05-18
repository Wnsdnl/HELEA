import os
import re
import json
import gzip
import bz2
import time
import codecs
import urllib.request
import urllib.parse
from collections import defaultdict, Counter
from typing import Any, Dict, List
from tqdm import tqdm


BASE_PATH = os.environ.get("BASE_PATH", "./data")  # set BASE_PATH env var or update this
DATA_PATH = f"{BASE_PATH}/1-hop_extraction/data"
INPUT_FILE = f"{BASE_PATH}/train/data/DW_negative.jsonl"

MAPPING_V1 = f"{BASE_PATH}/title_to_QID/title_to_qid_mapping.json"
MAPPING_V2 = f"{BASE_PATH}/title_to_QID/title_to_qid_mapping_ver2.json"
MAPPING_FILE_REL = f"{BASE_PATH}/relation_to_PID/relation_to_pid_mapping.json"
QID_LABEL_CACHE_FILE = f"{BASE_PATH}/title_to_QID/qid_to_label_cache.json"

KG_WIKI_DUMP = f"{DATA_PATH}/latest-truthy.nt.gz"
DBP_LABELS = f"{DATA_PATH}/labels_lang=en.ttl.bz2"
DBP_LITERALS = f"{DATA_PATH}/mappingbased-literals_lang=en.ttl.bz2"
DBP_OBJECTS = f"{DATA_PATH}/mappingbased-objects_lang=en.ttl.bz2"
DBP_INFOBOX = f"{DATA_PATH}/infobox-properties_lang=en.ttl.bz2" 
DBP_ONTOLOGY = f"{DATA_PATH}/ontology--DEV_type=parsed.nt"

OUTPUT_FILE = f"{BASE_PATH}/train/data/DW_negative_onlyEN.jsonl"

NOISE_RELATIONS = {
    "Link from a Wikipage to another Wikipage",
    "wasDerivedFrom",
    "is primary topic of",
    "page ID",
    "revision ID",
    "wikiPageID",
    "wikiPageRevisionID",
}

# 정규표현식
QID_FULL = re.compile(r"^Q\d+$")
QID_ANYWHERE = re.compile(r"Q\d+")
PAREN_AT_END_RE = re.compile(r"\s*\([^()]*\)\s*$")
LANG_TAG_RE = re.compile(r'^(.*?)(?:")?@([A-Za-z]{2,3}(?:-[A-Za-z0-9]+)*)$')

# ============================================================
# UTILS: 텍스트 정규화 (코드 1)
# ============================================================
def decode_unicode_escapes(s: str) -> str:
    if "\\u" in s or "\\U" in s or "\\x" in s:
        try: return codecs.decode(s, "unicode_escape")
        except Exception: return s
    return s

def strip_lang_tag(s: str) -> str:
    m = LANG_TAG_RE.match(s)
    return m.group(1) if m else s

def strip_paren_suffix(s: str) -> str:
    return PAREN_AT_END_RE.sub("", s).strip()

def normalize_text(s: str) -> str:
    s = decode_unicode_escapes(s)
    s = strip_lang_tag(s)
    s = strip_paren_suffix(s)
    return s.strip()

def normalize_obj(obj: Any) -> Any:
    if isinstance(obj, str): return normalize_text(obj)
    if isinstance(obj, list): return [normalize_obj(x) for x in obj]
    if isinstance(obj, dict): return {k: normalize_obj(v) for k, v in obj.items()}
    return obj

def clean_value(text: str) -> str:
    if not text: return ""
    text = re.sub(r'\"@\w+$', '', text)
    text = re.sub(r'\"\^\^<http.+>$', '', text)
    return text.strip('"').replace('_', ' ').strip()

def clean_uri(uri: str) -> str:
    return uri.split("/")[-1].split("#")[-1].replace("_", " ").strip()

def parse_ttl_line(line: str):
    try:
        if not line or line.startswith("#") or len(line.strip()) < 10: return None
        parts = line.split(" ", 2)
        if len(parts) < 3: return None
        s = parts[0].strip("<>")
        p = parts[1].strip("<>")
        o_raw = parts[2].rsplit(" .", 1)[0].strip()
        
        # 영어만 추출
        if '"@' in o_raw:
            if not o_raw.endswith('"@en'):
                return None
                
        o = o_raw.strip("<>") if o_raw.startswith("<") else clean_value(o_raw)
        return s, p, o
    except: return None

# ============================================================
# UTILS: QID Resolution & API (코드 2)
# ============================================================
def build_qid2bestname(name2qid: dict) -> Dict[str, str]:
    qid2best = {}
    for name, qid in name2qid.items():
        if not isinstance(name, str) or not isinstance(qid, str): continue
        prev = qid2best.get(qid)
        if prev is None or len(name) > len(prev):
            qid2best[qid] = name
    return qid2best

def iter_qids_from_json(obj) -> List[str]:
    found = []
    if isinstance(obj, dict):
        for v in obj.values(): found.extend(iter_qids_from_json(v))
    elif isinstance(obj, list):
        for v in obj: found.extend(iter_qids_from_json(v))
    elif isinstance(obj, str):
        if QID_FULL.match(obj): found.append(obj)
        else: found.extend(QID_ANYWHERE.findall(obj))
    return found

def replace_qids(obj, resolver):
    if isinstance(obj, dict):
        return {k: replace_qids(v, resolver) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_qids(x, resolver) for x in obj]
    if isinstance(obj, str):
        if QID_FULL.match(obj):
            r = resolver(obj)
            return r if r is not None else obj  # 해결되지 않으면 원본 그대로 반환
        def repl(m):
            q = m.group(0)
            r = resolver(q)
            return r if r is not None else q    # 해결되지 않으면 원본 그대로 반환
        return QID_ANYWHERE.sub(repl, obj)
    return obj

# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    print("=== Combined Pipeline: Extraction -> Resolution -> Normalization ===")

    # --------------------------------------------------------
    # [1] Load Mappings
    # --------------------------------------------------------
    print("[1/8] Loading mappings...")
    with open(MAPPING_V1, "r", encoding="utf-8") as f:
        name2qid_v1 = json.load(f)
    qid2name_v1 = build_qid2bestname(name2qid_v1)
    
    qid2name_v2 = {}
    if os.path.exists(MAPPING_V2):
        with open(MAPPING_V2, "r", encoding="utf-8") as f:
            qid2name_v2 = build_qid2bestname(json.load(f))
            
    with open(MAPPING_FILE_REL, "r", encoding="utf-8") as f:
        pid2label = {pid: label for label, pid in json.load(f).items()}

    try:
        with open(QID_LABEL_CACHE_FILE, "r", encoding="utf-8") as f:
            qid2label_cache = json.load(f)
    except FileNotFoundError:
        qid2label_cache = {}

    # --------------------------------------------------------
    # [2] Read ent_links
    # --------------------------------------------------------
    print("[2/8] Reading ent_links...")
    pairs, needed_dbp_urls, needed_wiki_qids = [], set(), set()
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # 파일 경로(INPUT_FILE)의 이름을 확인하여 label을 할당합니다.
            if "negative" in INPUT_FILE.lower():
                label = 0
            elif "positive" in INPUT_FILE.lower():
                label = 1
            else:
                label = -1 # 둘 다 없을 경우

            if "dbpedia_uri" in data and "wiki_qid" in data:
                left = data["dbpedia_uri"]
                raw_wiki = data["wiki_qid"]
                wd_qid = raw_wiki.split("/")[-1] if "/" in raw_wiki else raw_wiki
                
                best_wiki_name = qid2name_v1.get(wd_qid, qid2name_v2.get(wd_qid, qid2label_cache.get(wd_qid, wd_qid)))
                
                pairs.append({
                    "type": "db_wiki",
                    "uri_a": left,
                    "uri_b": wd_qid,
                    "entity_a": clean_uri(left),
                    "entity_b": best_wiki_name,
                    "label": label
                })
                needed_dbp_urls.add(left)
                needed_wiki_qids.add(wd_qid)

            elif "dbpedia_uri_1" in data and "dbpedia_uri_2" in data:
                left1 = data["dbpedia_uri_1"]
                left2 = data["dbpedia_uri_2"]
                
                pairs.append({
                    "type": "db_db",
                    "uri_a": left1,
                    "uri_b": left2,
                    "entity_a": clean_uri(left1),
                    "entity_b": clean_uri(left2),
                    "label": label
                })
                needed_dbp_urls.add(left1)
                needed_dbp_urls.add(left2)

            elif "wiki_qid_1" in data and "wiki_qid_2" in data:
                raw_wiki1 = data["wiki_qid_1"]
                raw_wiki2 = data["wiki_qid_2"]
                wd_qid1 = raw_wiki1.split("/")[-1] if "/" in raw_wiki1 else raw_wiki1
                wd_qid2 = raw_wiki2.split("/")[-1] if "/" in raw_wiki2 else raw_wiki2
                
                best_wiki_name1 = qid2name_v1.get(wd_qid1, qid2name_v2.get(wd_qid1, qid2label_cache.get(wd_qid1, wd_qid1)))
                best_wiki_name2 = qid2name_v1.get(wd_qid2, qid2name_v2.get(wd_qid2, qid2label_cache.get(wd_qid2, wd_qid2)))

                pairs.append({
                    "type": "wiki_wiki",
                    "uri_a": wd_qid1,
                    "uri_b": wd_qid2,
                    "entity_a": best_wiki_name1,
                    "entity_b": best_wiki_name2,
                    "label": label
                })
                needed_wiki_qids.add(wd_qid1)
                needed_wiki_qids.add(wd_qid2)
                
            else:
                print(f"!! [Line {line_num}] 알 수 없는 데이터 포맷 스킵: {data}")

    # --------------------------------------------------------
    # [3] DBpedia Ontology & Labels
    # --------------------------------------------------------
    print("[3/8] Building DBpedia rel_map and labels...")
    rel_map = {}
    if os.path.exists(DBP_ONTOLOGY):
        with open(DBP_ONTOLOGY, "r", encoding="utf-8") as f:
            for line in f:
                res = parse_ttl_line(line)
                if res and "label" in res[1]: rel_map[res[0]] = res[2]

    dbp_name_map = {}
    if os.path.exists(DBP_LABELS):
        with bz2.open(DBP_LABELS, "rt", encoding="utf-8") as f:
            for line in tqdm(f, desc="Scanning DBpedia Labels"):
                res = parse_ttl_line(line)
                if res and res[0] in needed_dbp_urls: dbp_name_map[res[0]] = res[2]

    # --------------------------------------------------------
    # [4] Extract DBpedia 1-hops
    # --------------------------------------------------------
    print("[4/8] Scanning DBpedia dumps (Objects, Literals, Infobox)...")
    kg_dbp = defaultdict(list)
    seen_dbp = defaultdict(set)
    for dump_path in [DBP_OBJECTS, DBP_LITERALS, DBP_INFOBOX]:
        if not os.path.exists(dump_path): continue
        with bz2.open(dump_path, "rt", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Scanning {os.path.basename(dump_path)}"):
                res = parse_ttl_line(line)
                if not res: continue
                s, p, o = res
                if s not in needed_dbp_urls: continue

                rel_label = rel_map.get(p, clean_uri(p))
                if rel_label in NOISE_RELATIONS: continue

                head_label = dbp_name_map.get(s, clean_uri(s))
                tail_label = clean_uri(o) if isinstance(o, str) and o.startswith("http") else (o if isinstance(o, str) else str(o))

                if (rel_label, tail_label) in seen_dbp[s]: continue
                seen_dbp[s].add((rel_label, tail_label))

                kg_dbp[s].append({"head": head_label, "relation": rel_label, "tail": tail_label})

    # --------------------------------------------------------
    # [5] Extract Wikidata 1-hops
    # --------------------------------------------------------
    print("[5/8] Scanning Wikidata dump...")
    kg_wiki = defaultdict(list)
    if os.path.exists(KG_WIKI_DUMP):
        with gzip.open(KG_WIKI_DUMP, "rt", encoding="utf-8") as f:
            for line in tqdm(f, desc="Scanning Wiki"):
                parts = line.split(" ", 2)
                if len(parts) < 3: continue
                sub_qid = parts[0].strip("<>").split("/")[-1]
                if sub_qid in needed_wiki_qids:
                    pred_url = parts[1].strip("<>")
                    pid = pred_url.split("/")[-1]
                    rel_text = pid2label.get(pid) or (pred_url.split("schema.org/")[-1] if "schema.org" in pred_url else None)
                    if not rel_text: continue
                    
                    obj_raw = parts[2].rsplit(" .", 1)[0].strip()
                    if "/entity/Q" in obj_raw:
                        obj_qid = obj_raw.strip("<>").split("/")[-1]
                        
                        head_name = qid2name_v1.get(sub_qid, qid2name_v2.get(sub_qid, sub_qid))
                        tail_name = qid2name_v1.get(obj_qid, qid2name_v2.get(obj_qid, obj_qid))
                        
                        kg_wiki[sub_qid].append({"head": head_name, "relation": rel_text, "tail": tail_name})

    # --------------------------------------------------------
    # [6] Assemble Raw Data & Identify Missing QIDs
    # --------------------------------------------------------
    print("[6/8] Assembling raw rows and identifying unresolved QIDs...")
    raw_rows = []
    cnt = Counter()
    for item in pairs:
        if item["type"] == "db_wiki":
            hops_a = kg_dbp.get(item["uri_a"], [])
            hops_b = kg_wiki.get(item["uri_b"], [])
        elif item["type"] == "db_db":
            hops_a = kg_dbp.get(item["uri_a"], [])
            hops_b = kg_dbp.get(item["uri_b"], [])
        elif item["type"] == "wiki_wiki":
            hops_a = kg_wiki.get(item["uri_a"], [])
            hops_b = kg_wiki.get(item["uri_b"], [])
            
        row = {
            "entity_a": item["entity_a"],
            "1-hops_a": hops_a,
            "entity_b": item["entity_b"],
            "1-hops_b": hops_b,
            "label": item["label"]
        }
        raw_rows.append(row)
        for q in iter_qids_from_json(row):
            cnt[q] += 1

    all_qids = list(cnt.keys())
    need_api = [q for q in all_qids if q not in qid2name_v1 and q not in qid2name_v2 and q not in qid2label_cache]
    print(f"Total unique QIDs found: {len(all_qids):,}")
    print(f"원본 그대로 유지되는(API 호출을 생략한) QID 개수: {len(need_api):,}")

    # --------------------------------------------------------
    # [7] Fetch Missing QIDs via API (수정됨: 호출 완전 제외)
    # --------------------------------------------------------
    print("[7/8] API Fetching bypassed as requested.")

    # --------------------------------------------------------
    # [8] Replace QIDs, Normalize, and Save
    # --------------------------------------------------------
    print("[8/8] Applying QID replacement, Normalization, and Saving...")
    
    def qid_resolver(qid: str):
        return qid2name_v1.get(qid) or qid2name_v2.get(qid) or qid2label_cache.get(qid)

    valid_count = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in tqdm(raw_rows, desc="Final Processing"):
            # 1. QID 치환 (해결되지 않은 QID는 원본 유지됨)
            resolved_row = replace_qids(row, qid_resolver)
            # 2. 텍스트 정규화
            normalized_row = normalize_obj(resolved_row)
            
            f.write(json.dumps(normalized_row, ensure_ascii=False) + "\n")
            valid_count += 1

    print(f"=== Done! Saved {valid_count} fully processed rows to {OUTPUT_FILE} ===")

if __name__ == "__main__":
    main()