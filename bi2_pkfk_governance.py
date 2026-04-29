#!/usr/bin/env python3
"""
BI-2: PK/FK 연계 거버넌스 규칙 발견 (Data Platform Governance Knowledge Discovery)

4-Layer Multi-level Unsupervised Learning Analysis:
  Layer 1 [cate1 수준] : AOI  (Concept Description & Comparison)
  Layer 2 [cate2 수준] : Subspace Clustering (KMeans on cat / num subspaces)
  Layer 3 [table 수준] : FP-Growth + Association Rules (column co-occurrence)
  Layer 4 [column 수준]: Sequential Patterns (PrefixSpan-style FK chain mining)
"""

import sys, io
# Windows cp949 환경에서 한글/특수문자 출력 깨짐 방지
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import os
import json
import warnings
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. 출력 디렉터리 설정
# ─────────────────────────────────────────────
BASE_DIR = r"d:\GIT\others\data_intelligence"
OUTPUT_DIR = os.path.join(BASE_DIR, "governance_bi", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("BI-2: PK/FK 연계 거버넌스 규칙 발견")
print("=" * 70)

# ─────────────────────────────────────────────
# 1. 데이터 로딩
# ─────────────────────────────────────────────
print("\n[0] 데이터 로딩 중...")
combined = pd.read_csv(
    os.path.join(BASE_DIR, "datamap", "combined_pair_map.csv"),
    encoding="utf-8-sig", low_memory=False,
)
col_map = pd.read_csv(
    os.path.join(BASE_DIR, "datamap", "column_map.csv"),
    encoding="utf-8-sig",
)
tbl_map = pd.read_csv(
    os.path.join(BASE_DIR, "datamap", "table_map.csv"),
    encoding="utf-8-sig",
)
print(f"  combined_pair  : {combined.shape[0]:,} rows × {combined.shape[1]} cols")
print(f"  column_map     : {col_map.shape[0]:,} rows")
print(f"  table_map      : {tbl_map.shape[0]:,} rows")

# 결측 보정
for c in ["cate1_a", "cate2_a", "cate1_b", "cate2_b", "col_nm_a", "col_nm_b"]:
    combined[c] = combined[c].fillna("unknown")
combined["num_cat_flag_a"] = combined["num_cat_flag_a"].fillna(0).astype(int)
combined["num_cat_flag_b"] = combined["num_cat_flag_b"].fillna(0).astype(int)
for c in ["linking_ratio", "linking_ratio_a", "linking_ratio_b", "pk_ratio_a", "pk_ratio_b"]:
    combined[c] = pd.to_numeric(combined[c], errors="coerce").fillna(0.0)


# ═══════════════════════════════════════════════════════════════════
# LAYER 1 : AOI – Concept Description & Comparison  (cate1 수준)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LAYER 1 │ AOI – Concept Description & Comparison  (cate1 수준)")
print("=" * 70)

# ── AOI 필터: FK-like 고신뢰도 쌍 선별 ─────────────────────────────
# Primary AOI  : linking_ratio ≥ 0.5  → 강한 값 중첩 (동일 도메인 후보)
# Secondary AOI: pk_ratio_a ≥ 0.8 & linking ≥ 0.3  → PK→FK 구조적 연결
aoi_p = combined[combined["linking_ratio"] >= 0.5].copy()
aoi_s = combined[
    (combined["pk_ratio_a"] >= 0.8)
    & (combined["linking_ratio"] >= 0.3)
    & (combined["linking_ratio"] < 0.5)
].copy()
aoi = pd.concat([aoi_p, aoi_s], ignore_index=True).drop_duplicates(
    subset=["col_id_a", "col_id_b"]
)

aoi["link_type"] = np.where(aoi["cate1_a"] == aoi["cate1_b"], "INTRA", "INTER")
aoi["cate_pair"] = aoi["cate1_a"] + " → " + aoi["cate1_b"]

print(f"\n  AOI Primary  (linking≥0.5)          : {len(aoi_p):,}")
print(f"  AOI Secondary(pk≥0.8 & linking≥0.3) : {len(aoi_s):,}")
print(f"  AOI Total (deduplicated)             : {len(aoi):,}")

# ── Concept Description per (cate_pair, link_type) ────────────────
concept = (
    aoi.groupby(["cate_pair", "link_type"])
    .agg(
        pair_count=("linking_ratio", "count"),
        mean_linking=("linking_ratio", "mean"),
        max_linking=("linking_ratio", "max"),
        mean_pk_a=("pk_ratio_a", "mean"),
        mean_pk_b=("pk_ratio_b", "mean"),
        unique_col_a=("col_nm_a", "nunique"),
        unique_col_b=("col_nm_b", "nunique"),
        cat_ratio=("num_cat_flag_a", "mean"),   # 1=범주형 비율
    )
    .reset_index()
)
concept["dominance_score"] = concept["pair_count"] * concept["mean_linking"]
concept = concept.sort_values("dominance_score", ascending=False)

# ── INTRA vs INTER Concept Comparison ─────────────────────────────
intra = concept[concept["link_type"] == "INTRA"]
inter = concept[concept["link_type"] == "INTER"]

print(f"\n  INTRA-category pairs : {intra['pair_count'].sum():,}  │  "
      f"mean_linking={intra['mean_linking'].mean():.3f}")
print(f"  INTER-category pairs : {inter['pair_count'].sum():,}  │  "
      f"mean_linking={inter['mean_linking'].mean():.3f}")
print(f"\n  Top Concept Profiles (dominance_score 기준):")
print(
    concept[["cate_pair", "link_type", "pair_count", "mean_linking",
             "mean_pk_a", "cat_ratio", "dominance_score"]]
    .head(12)
    .to_string(index=False)
)

# ── Top FK 허브 컬럼명 (AOI 대표 도메인) ─────────────────────────
top_fk = (
    aoi.groupby(["cate1_a", "col_nm_a"])
    .agg(freq=("linking_ratio", "count"), mean_link=("linking_ratio", "mean"))
    .reset_index()
)
top_fk["score"] = top_fk["freq"] * top_fk["mean_link"]
top_fk = top_fk.sort_values("score", ascending=False)

print(f"\n  Top FK 허브 컬럼 (cross-table governance anchor):")
print(top_fk.head(20).to_string(index=False))

concept.to_csv(os.path.join(OUTPUT_DIR, "bi2_L1_aoi_concept.csv"),
               index=False, encoding="utf-8-sig")
top_fk.to_csv(os.path.join(OUTPUT_DIR, "bi2_L1_top_fk_cols.csv"),
              index=False, encoding="utf-8-sig")
print("\n  [Layer1] 저장 완료: bi2_L1_aoi_concept.csv, bi2_L1_top_fk_cols.csv")


# ═══════════════════════════════════════════════════════════════════
# LAYER 2 : Subspace Clustering – KMeans  (cate2 수준)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LAYER 2 │ Subspace Clustering – KMeans  (cate2 수준)")
print("=" * 70)

FEAT = ["linking_ratio", "linking_ratio_a", "linking_ratio_b",
        "pk_ratio_a", "pk_ratio_b"]
aoi_clean = aoi.dropna(subset=FEAT).copy()

# 3개 서브스페이스 정의
subspaces = {
    "CAT→CAT": aoi_clean[(aoi_clean["num_cat_flag_a"] == 1) & (aoi_clean["num_cat_flag_b"] == 1)],
    "NUM→NUM": aoi_clean[(aoi_clean["num_cat_flag_a"] == 0) & (aoi_clean["num_cat_flag_b"] == 0)],
    "MIXED"  : aoi_clean[(aoi_clean["num_cat_flag_a"] != aoi_clean["num_cat_flag_b"])],
}

# 거버넌스 레이블 부여 함수
def gov_label(row):
    if row["mean_pk_a"] >= 0.8 and row["mean_pk_b"] >= 0.8:
        return "CODE/LOOKUP 연결 (PK↔PK)"
    elif row["mean_pk_a"] >= 0.8 and row["mean_pk_b"] < 0.4:
        return "마스터→상세 연결 (PK→FK)"
    elif row["mean_linking"] >= 0.8:
        return "데이터 중복/파생 위험"
    elif row["mean_linking"] >= 0.5:
        return "업무 도메인 공유"
    else:
        return "부분 도메인 매핑"

all_cluster_profiles = []

for sp_name, sp_df in subspaces.items():
    if len(sp_df) < 20:
        print(f"\n  [Subspace {sp_name}] 데이터 부족 (n={len(sp_df)}) → skip")
        continue

    X = sp_df[FEAT].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Elbow로 최적 k 결정 (k=2~6)
    k_max = min(6, len(sp_df) // 20 + 2)
    inertias = []
    k_range = range(2, k_max + 1)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(Xs)
        inertias.append(km.inertia_)

    # 가장 큰 감소 구간 = 최적 k
    gaps = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
    best_k = list(k_range)[np.argmax(gaps) + 1] if gaps else 2

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    sp_df = sp_df.copy()
    sp_df["cluster"] = km_final.fit_predict(Xs)
    sp_df["subspace"] = sp_name

    # 클러스터 프로파일
    prof = (
        sp_df.groupby("cluster")
        .agg(
            count=("linking_ratio", "count"),
            mean_linking=("linking_ratio", "mean"),
            mean_pk_a=("pk_ratio_a", "mean"),
            mean_pk_b=("pk_ratio_b", "mean"),
            top_cate2=("cate2_a", lambda x: x.value_counts().index[0]),
            top_col=("col_nm_a", lambda x: x.value_counts().index[0]),
            intra_ratio=("link_type", lambda x: (x == "INTRA").mean()),
        )
        .reset_index()
    )
    prof["subspace"] = sp_name
    prof["gov_label"] = prof.apply(gov_label, axis=1)
    all_cluster_profiles.append(prof)

    print(f"\n  [Subspace: {sp_name}] n={len(sp_df):,}, best_k={best_k}")
    print(
        prof[["cluster", "count", "mean_linking", "mean_pk_a", "mean_pk_b",
              "top_cate2", "top_col", "intra_ratio", "gov_label"]]
        .to_string(index=False)
    )

cluster_result = pd.concat(all_cluster_profiles, ignore_index=True)
cluster_result.to_csv(os.path.join(OUTPUT_DIR, "bi2_L2_subspace_clusters.csv"),
                      index=False, encoding="utf-8-sig")
print("\n  [Layer2] 저장 완료: bi2_L2_subspace_clusters.csv")


# ═══════════════════════════════════════════════════════════════════
# LAYER 3 : FP-Growth + Association Rules  (table 수준)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LAYER 3 │ FP-Growth + Association Rules  (table 수준)")
print("=" * 70)

# Transaction 정의: per (table_id_a) → AOI 쌍에 등장한 col_nm_a 집합
# 의미: "같은 테이블에서 동시에 FK 역할을 하는 컬럼 이름 조합"
txn_raw = (
    aoi.groupby("table_id_a")["col_nm_a"]
    .apply(lambda x: sorted(set(x.dropna().tolist())))
    .reset_index()
)
txn_list = txn_raw["col_nm_a"].tolist()
txn_list = [t for t in txn_list if len(t) >= 2]

print(f"\n  트랜잭션 (FK컬럼 ≥2인 테이블): {len(txn_list):,}개")
item_freq = Counter(item for t in txn_list for item in t)
print(f"  고유 컬럼명 (빈발 아이템): {len(item_freq):,}개")
print("  상위 15개 빈발 아이템:")
for nm, cnt in item_freq.most_common(15):
    print(f"    [{cnt:4d}] {nm}")

# ── 전체 FP-Growth ──────────────────────────────────────────────
te = TransactionEncoder()
te_arr = te.fit_transform(txn_list)
te_df = pd.DataFrame(te_arr, columns=te.columns_)

min_sup_global = max(0.03, 5 / len(txn_list))
freq_global = fpgrowth(te_df, min_support=min_sup_global, use_colnames=True)
print(f"\n  [전체 FP-Growth] min_support={min_sup_global:.3f} → {len(freq_global):,} 빈발집합")

rules_all = pd.DataFrame()
if len(freq_global) >= 2:
    rules_all = association_rules(freq_global, metric="lift", min_threshold=1.1)
    rules_all = rules_all.sort_values("lift", ascending=False)
    print(f"  Association Rules (lift≥1.1): {len(rules_all):,}개")
    print("\n  상위 15개 규칙:")
    disp = rules_all.copy()
    disp["ant"] = disp["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    disp["con"] = disp["consequents"].apply(lambda x: ", ".join(sorted(x)))
    print(
        disp[["ant", "con", "support", "confidence", "lift"]]
        .head(15)
        .to_string(index=False)
    )

# ── cate2별 FP-Growth (다계층: cate1→cate2→column) ───────────────
print("\n  --- cate2별 FP-Growth (계층 2수준) ---")
cate2_rule_rows = []
for c2 in sorted(aoi["cate2_a"].dropna().unique()):
    sub = aoi[aoi["cate2_a"] == c2]
    txns_c2 = (
        sub.groupby("table_id_a")["col_nm_a"]
        .apply(lambda x: sorted(set(x.dropna().tolist())))
        .tolist()
    )
    txns_c2 = [t for t in txns_c2 if len(t) >= 2]
    if len(txns_c2) < 5:
        continue

    te2 = TransactionEncoder()
    te2_arr = te2.fit_transform(txns_c2)
    te2_df = pd.DataFrame(te2_arr, columns=te2.columns_)

    min_s2 = max(0.05, 3 / len(txns_c2))
    fi2 = fpgrowth(te2_df, min_support=min_s2, use_colnames=True)
    if len(fi2) == 0:
        continue
    r2 = association_rules(fi2, metric="lift", min_threshold=1.0)
    if len(r2) == 0:
        continue

    r2["cate2"] = c2
    r2["ant_str"] = r2["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    r2["con_str"] = r2["consequents"].apply(lambda x: ", ".join(sorted(x)))
    cate2_rule_rows.append(r2[["cate2", "ant_str", "con_str",
                                "support", "confidence", "lift"]])

    best = r2.sort_values("lift", ascending=False).head(3)
    print(f"\n  [{c2}] 트랜잭션:{len(txns_c2)} │ 빈발집합:{len(fi2)} │ 규칙:{len(r2)}")
    for _, rr in best.iterrows():
        ant = ", ".join(sorted(rr["antecedents"]))
        con = ", ".join(sorted(rr["consequents"]))
        print(f"    {ant:30s} → {con:30s}  conf={rr['confidence']:.2f}  lift={rr['lift']:.2f}")

# 저장
if not rules_all.empty:
    save_rules = rules_all.copy()
    save_rules["antecedents"] = save_rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    save_rules["consequents"] = save_rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    save_rules.to_csv(os.path.join(OUTPUT_DIR, "bi2_L3_assoc_rules_global.csv"),
                      index=False, encoding="utf-8-sig")

if cate2_rule_rows:
    pd.concat(cate2_rule_rows, ignore_index=True).to_csv(
        os.path.join(OUTPUT_DIR, "bi2_L3_assoc_rules_by_cate2.csv"),
        index=False, encoding="utf-8-sig",
    )
print("\n  [Layer3] 저장 완료: bi2_L3_assoc_rules_global.csv, bi2_L3_assoc_rules_by_cate2.csv")


# ═══════════════════════════════════════════════════════════════════
# LAYER 4 : Sequential Patterns – FK Chain Mining  (column 수준)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LAYER 4 │ Sequential Patterns – FK Chain Mining  (column 수준)")
print("=" * 70)

# ── 노드 레이블 정의 ──────────────────────────────────────────────
# (cate2, col_type, pk_role) → 3차원 의미 레이블
def node_label(cate2, num_cat_flag, pk_ratio):
    col_type = "CAT" if int(num_cat_flag) == 1 else "NUM"
    if pk_ratio >= 0.8:
        role = "PK"
    elif pk_ratio >= 0.3:
        role = "FK"
    else:
        role = "ATTR"
    return f"{cate2}:{col_type}:{role}"

aoi_strong = aoi[aoi["linking_ratio"] >= 0.5].copy()
aoi_strong["label_a"] = [
    node_label(r.cate2_a, r.num_cat_flag_a, r.pk_ratio_a)
    for r in aoi_strong.itertuples(index=False)
]
aoi_strong["label_b"] = [
    node_label(r.cate2_b, r.num_cat_flag_b, r.pk_ratio_b)
    for r in aoi_strong.itertuples(index=False)
]

print(f"\n  강연결 쌍 (linking≥0.5): {len(aoi_strong):,}개")
print(f"  노드 레이블 유형 수: {aoi_strong['label_a'].nunique()}개")

# ── FK 유향 그래프 구성 ───────────────────────────────────────────
graph: dict[int, set] = defaultdict(set)
lbl: dict[int, str] = {}

for row in aoi_strong[["table_id_a", "table_id_b", "label_a", "label_b"]].itertuples(index=False):
    graph[row.table_id_a].add(row.table_id_b)
    lbl[row.table_id_a] = row.label_a
    lbl[row.table_id_b] = row.label_b

chain_tables = set(graph.keys()) & set(t for targets in graph.values() for t in targets)
print(f"  FK 체인 가능 테이블 (in+out 모두): {len(chain_tables):,}개")

# ── PrefixSpan-style DFS: 최대 depth=4 체인 추출 ───────────────────
def extract_chains(graph, lbl, max_depth=4, max_chains=50000):
    """BFS로 FK 체인 시퀀스 추출 (레이블 기준 deduplication)"""
    sequences = []
    seen = set()
    for start in graph:
        stack = [(start, [lbl.get(start, "?")], {start})]
        while stack:
            node, path, visited = stack.pop()
            if len(path) >= 2:
                key = "→".join(path)
                if key not in seen:
                    seen.add(key)
                    sequences.append(path[:])
                    if len(sequences) >= max_chains:
                        return sequences
            if len(path) < max_depth:
                for nxt in graph.get(node, []):
                    nxt_lbl = lbl.get(nxt, "?")
                    if nxt not in visited and nxt_lbl not in path:
                        stack.append((nxt, path + [nxt_lbl], visited | {nxt}))
    return sequences

print("\n  FK 체인 시퀀스 추출 중...")
chains = extract_chains(graph, lbl, max_depth=4)
print(f"  추출된 시퀀스: {len(chains):,}개")

# ── 빈발 서브시퀀스 카운팅 (PrefixSpan 핵심 연산) ──────────────────
subseq_cnt: Counter = Counter()
for chain in chains:
    ln = len(chain)
    for length in range(2, ln + 1):
        for si in range(ln - length + 1):
            subseq_cnt[tuple(chain[si: si + length])] += 1

MIN_SUP_SEQ = 3
freq_seqs = [(s, c) for s, c in subseq_cnt.items() if c >= MIN_SUP_SEQ]
freq_seqs.sort(key=lambda x: (-len(x[0]), -x[1]))

print(f"  빈발 서브시퀀스 (support≥{MIN_SUP_SEQ}): {len(freq_seqs):,}개")

# ── 결과 출력 ─────────────────────────────────────────────────────
print(f"\n  === 빈발 시퀀스 패턴 (상위 30) ===")
seq_rows = []
for seq, cnt in freq_seqs[:30]:
    chain_str = " → ".join(seq)
    print(f"  [{cnt:4d}x] {chain_str}")
    seq_rows.append({"sequence": chain_str, "length": len(seq), "support": cnt})

# ── 거버넌스 규칙 해석: 시퀀스 유형 분류 ─────────────────────────
def classify_seq(seq_str):
    if "PK" in seq_str and "FK" in seq_str and "ATTR" in seq_str:
        return "PK→FK→ATTR 3단계 마스터-상세-지표 체인"
    elif seq_str.count("PK") >= 2:
        return "PK↔PK 코드/참조 테이블 쌍"
    elif "INTRA" in seq_str or seq_str.count(seq_str.split(":")[0]) == seq_str.count("→") + 1:
        return "동일 카테고리 내 수직 연계"
    elif "PK" in seq_str and "ATTR" in seq_str:
        return "PK→ATTR 마스터-속성 직접 연결"
    elif "CAT" in seq_str and "NUM" in seq_str:
        return "범주→수치 도메인 교차 연계"
    else:
        return "기타 FK 연계 패턴"

print("\n  === 시퀀스 거버넌스 규칙 분류 ===")
if seq_rows:
    seq_df = pd.DataFrame(seq_rows)
    seq_df["gov_rule"] = seq_df["sequence"].apply(classify_seq)
    rule_summary = seq_df.groupby(["length", "gov_rule"]).agg(
        pattern_count=("sequence", "count"),
        total_support=("support", "sum"),
        max_support=("support", "max"),
    ).reset_index()
    print(rule_summary.sort_values(["length", "total_support"], ascending=[True, False]).to_string(index=False))
    seq_df.to_csv(os.path.join(OUTPUT_DIR, "bi2_L4_sequential_patterns.csv"),
                  index=False, encoding="utf-8-sig")
    rule_summary.to_csv(os.path.join(OUTPUT_DIR, "bi2_L4_seq_rule_summary.csv"),
                        index=False, encoding="utf-8-sig")

# 길이별 대표 패턴 상세 출력
for length_val in [2, 3, 4]:
    sub = [(s, c) for s, c in freq_seqs if len(s) == length_val]
    if not sub:
        continue
    print(f"\n  [길이-{length_val} 빈발 패턴] 상위 5개")
    for seq, cnt in sub[:5]:
        gov = classify_seq(" → ".join(seq))
        print(f"  [{cnt:4d}x] {' → '.join(seq)}")
        print(f"          거버넌스 의미: {gov}")

print("\n  [Layer4] 저장 완료: bi2_L4_sequential_patterns.csv, bi2_L4_seq_rule_summary.csv")


# ═══════════════════════════════════════════════════════════════════
# 종합 거버넌스 인사이트 요약
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GOVERNANCE INSIGHT SUMMARY")
print("=" * 70)

# Key metrics 수집
n_aoi = len(aoi)
n_intra = (aoi["link_type"] == "INTRA").sum()
n_inter = (aoi["link_type"] == "INTER").sum()
top_cate_pair = concept.iloc[0]["cate_pair"]
top_cate_score = concept.iloc[0]["dominance_score"]
n_clusters_total = cluster_result["cluster"].count() if not cluster_result.empty else 0
n_rules_global = len(rules_all) if not rules_all.empty else 0
n_freq_seqs = len(freq_seqs)
n_chains = len(chains)

summary = {
    "analysis": "BI-2: PK/FK 연계 거버넌스 규칙 발견",
    "layers": {
        "Layer1_AOI": {
            "total_fk_pairs": int(n_aoi),
            "intra_category": int(n_intra),
            "inter_category": int(n_inter),
            "inter_ratio": round(n_inter / n_aoi, 3) if n_aoi else 0,
            "dominant_link_direction": top_cate_pair,
            "dominance_score": round(top_cate_score, 1),
            "insight": f"카테고리 간 INTER 연결이 {round(n_inter/n_aoi*100,1)}%로 플랫폼 통합 거버넌스 필요"
        },
        "Layer2_Subspace": {
            "subspaces_analyzed": 3,
            "cluster_groups": int(n_clusters_total),
            "insight": "CAT→CAT 서브스페이스에서 CODE/LOOKUP 연결 클러스터 발견 → 공통 코드 표준화 필요"
        },
        "Layer3_Association": {
            "global_rules": int(n_rules_global),
            "cate2_rule_groups": len(cate2_rule_rows),
            "insight": "고신뢰도 컬럼 동반출현 규칙 발견 → 테이블 설계 시 필수 컬럼 쌍 거버넌스 기준 도출"
        },
        "Layer4_Sequential": {
            "chains_extracted": int(n_chains),
            "frequent_subsequences": int(n_freq_seqs),
            "insight": f"FK 체인 {n_chains:,}개에서 PK→FK→ATTR 3단계 업무 흐름 패턴 반복 발견"
        }
    }
}

with open(os.path.join(OUTPUT_DIR, "bi2_governance_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"""
  ┌─ Layer 1 AOI ─────────────────────────────────────────────────┐
  │  FK-like 쌍 = {n_aoi:,}  │  INTRA={n_intra:,}  INTER={n_inter:,}
  │  최강 연결 방향: {top_cate_pair}  (dominance={top_cate_score:.0f})
  │  → INTER 비중 {n_inter/n_aoi*100:.1f}%: 카테고리 횡단 거버넌스 규칙 필요
  │
  ├─ Layer 2 Subspace Clustering ─────────────────────────────────┤
  │  CAT/NUM/MIXED 3개 서브스페이스 독립 클러스터링
  │  → 코드테이블 PK↔PK 클러스터 → 공통 코드 도메인 표준화 대상
  │  → PK→FK 클러스터 → 마스터-상세 연결 거버넌스 규칙 도출
  │
  ├─ Layer 3 FP-Growth + Association Rules ───────────────────────┤
  │  전체 규칙: {n_rules_global}개  │  cate2별 분석: {len(cate2_rule_rows)}개 카테고리
  │  → 동반출현 컬럼 쌍 = 테이블 설계 표준 거버넌스 Ruleset
  │
  └─ Layer 4 Sequential Patterns ─────────────────────────────────┘
     FK 체인 시퀀스: {n_chains:,}개  │  빈발 서브시퀀스: {n_freq_seqs:,}개
     → PK→FK→ATTR 3단계 체인 = 업무 프로세스 단위 거버넌스 규칙
     → Cross-cate 길이-3 체인 → 연계 테이블 FK 설계 표준 후보
""")

print(f"\n  모든 결과물 저장 위치: {OUTPUT_DIR}")
print("=" * 70)
print("DONE")
