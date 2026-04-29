# -*- coding: utf-8 -*-
import sys, os, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import json

# ── 경로 상수 ──────────────────────────────────────────────────────────────
RAW = r'd:\GIT\others\data_intelligence\bi_rawdata'
OUT = r'd:\GIT\others\data_intelligence\governance_bi\bi3_output'
os.makedirs(OUT, exist_ok=True)

F_QUIT   = os.path.join(RAW, '한국건강증진개발원_국가금연지원서비스 등록정보(기본)_20241231.csv')
F_HEART  = os.path.join(RAW, '한국건강증진개발원_보건소 모바일 헬스케어_심박수_20251120.csv')
F_VISIT  = os.path.join(RAW, '한국건강증진개발원_찾아가는 금연지원서비스 방문연계기관 정보_20241231.csv')
F_GOODS  = os.path.join(RAW, '한국건강증진개발원_국가금연지원서비스 물품 구매현황_20241231.csv')
F_POLICY = os.path.join(RAW, '한국건강증진개발원_국민건강증진종합계획_만성퇴행성질환과 발병위험 요인관리 지표_20201231.csv')
F_CHRONIC= os.path.join(RAW, '경기도 화성시_만성질환관리 프로그램 통계(중재)_20191231.csv')

sep_line = "=" * 72

def header(title):
    print(f"\n{sep_line}")
    print(f"  {title}")
    print(sep_line)

# ══════════════════════════════════════════════════════════════════════════════
# 0. 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════
header("0. 데이터 로드")

df_quit   = pd.read_csv(F_QUIT,   encoding='cp949', low_memory=False)
df_heart  = pd.read_csv(F_HEART,  encoding='cp949', low_memory=False)
df_visit  = pd.read_csv(F_VISIT,  encoding='cp949', low_memory=False)
df_goods  = pd.read_csv(F_GOODS,  encoding='cp949', low_memory=False)
df_policy = pd.read_csv(F_POLICY, encoding='utf-8-sig', low_memory=False)
df_chronic= pd.read_csv(F_CHRONIC, encoding='cp949', low_memory=False)

print(f"금연서비스 등록정보 : {df_quit.shape}")
print(f"심박수             : {df_heart.shape}")
print(f"방문연계기관       : {df_visit.shape}")
print(f"물품구매현황       : {df_goods.shape}")
print(f"건강증진종합계획   : {df_policy.shape}")
print(f"만성질환관리 통계  : {df_chronic.shape}")

# ── 컬럼명 공백 제거
for df in [df_quit, df_heart, df_visit, df_goods, df_policy, df_chronic]:
    df.columns = df.columns.str.strip()

print("\n[금연서비스 컬럼]:", list(df_quit.columns))
print("[건강증진계획 컬럼]:", list(df_policy.columns))
print("[만성질환 컬럼]:", list(df_chronic.columns))
print("[물품구매 컬럼]:", list(df_goods.columns))
print("[방문연계 컬럼]:", list(df_visit.columns))

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 : AOI Concept Description & Comparison
# ══════════════════════════════════════════════════════════════════════════════
header("LAYER 1-A : AOI — 국민건강증진종합계획 중점과제별 달성도 개념 기술")

# 달성도/현재값/목표값 관련 컬럼 탐색
print("[건강증진계획 전체 컬럼]")
for c in df_policy.columns:
    print(f"  '{c}'")

# 중점과제명 컬럼 찾기
task_col = None
for c in df_policy.columns:
    if '중점' in c or '과제' in c or '분야' in c:
        task_col = c
        break
if task_col is None:
    task_col = df_policy.columns[0]
print(f"\n중점과제 컬럼: '{task_col}'")
print(df_policy[task_col].value_counts().head(20))

# 수치형 컬럼 찾기 (기준값, 목표값, 현재값, 달성도)
num_cols_policy = df_policy.select_dtypes(include='number').columns.tolist()
print(f"\n수치형 컬럼: {num_cols_policy}")

# 달성도 컬럼 탐색
achieve_col = None
for c in df_policy.columns:
    if '달성' in c:
        achieve_col = c
        break
current_col = None
for c in df_policy.columns:
    if '현재' in c:
        current_col = c
        break
target_col = None
for c in df_policy.columns:
    if '목표' in c:
        target_col = c
        break
base_col = None
for c in df_policy.columns:
    if '기준' in c:
        base_col = c
        break

print(f"\n달성도컬럼='{achieve_col}', 현재값='{current_col}', 목표값='{target_col}', 기준값='{base_col}'")

# 현황(달성/개선/유지/악화) 컬럼
status_col = None
for c in df_policy.columns:
    if '현황' in c or '상태' in c:
        status_col = c
        break
print(f"현황컬럼='{status_col}'")

# ── AOI 개념기술: 중점과제명 × 달성 현황 분포
if status_col and task_col:
    df_policy[status_col] = df_policy[status_col].astype(str).str.strip()
    df_policy[task_col]   = df_policy[task_col].astype(str).str.strip()
    aoi_policy = df_policy.groupby([task_col, status_col]).size().reset_index(name='count')
    total_by_task = df_policy.groupby(task_col).size().reset_index(name='total')
    aoi_policy = aoi_policy.merge(total_by_task, on=task_col)
    aoi_policy['ratio'] = (aoi_policy['count'] / aoi_policy['total']).round(4)
    print("\n[AOI 개념기술: 중점과제 × 달성현황 비율]")
    print(aoi_policy.to_string(index=False))
    aoi_policy.to_csv(os.path.join(OUT, 'bi3_L1a_policy_aoi.csv'), index=False, encoding='utf-8-sig')
    print(f"\n>> 저장: bi3_L1a_policy_aoi.csv")

# ── AOI: 달성률 분포 통계 by 중점과제
if achieve_col:
    df_policy[achieve_col] = pd.to_numeric(df_policy[achieve_col], errors='coerce')
    aoi_achieve = df_policy.groupby(task_col)[achieve_col].agg(['mean','std','min','max','count']).reset_index()
    aoi_achieve.columns = [task_col, 'mean_achievement', 'std', 'min', 'max', 'n']
    aoi_achieve = aoi_achieve.sort_values('mean_achievement', ascending=False)
    print("\n[AOI: 중점과제별 달성도 통계]")
    print(aoi_achieve.to_string(index=False))
    aoi_achieve.to_csv(os.path.join(OUT, 'bi3_L1a_achieve_stats.csv'), index=False, encoding='utf-8-sig')

header("LAYER 1-B : AOI — 만성질환 관리 프로그램 질환별 참여 강도 비교")
print("[만성질환 컬럼 상세]")
for c in df_chronic.columns:
    print(f"  '{c}' : {df_chronic[c].dtype} | sample: {df_chronic[c].dropna().unique()[:5]}")

# 질환/성별 컬럼 탐색
dis_col = None
for c in df_chronic.columns:
    if '질환' in c or '병명' in c:
        dis_col = c
        break
sex_col_chr = None
for c in df_chronic.columns:
    if '성별' in c or '성' == c:
        sex_col_chr = c
        break
age_col = None
for c in df_chronic.columns:
    if '연령' in c or '나이' in c:
        age_col = c
        break

print(f"\n질환컬럼='{dis_col}', 성별='{sex_col_chr}', 연령='{age_col}'")

# 참여 강도 수치 컬럼
num_chr = df_chronic.select_dtypes(include='number').columns.tolist()
print(f"수치형 컬럼: {num_chr}")

if dis_col and num_chr:
    group_cols = [c for c in [dis_col, sex_col_chr, age_col] if c is not None]
    aoi_chronic = df_chronic.groupby(group_cols)[num_chr].mean().round(2).reset_index()
    print("\n[AOI: 질환별 프로그램 참여 강도 평균]")
    print(aoi_chronic.to_string(index=False))
    aoi_chronic.to_csv(os.path.join(OUT, 'bi3_L1b_chronic_aoi.csv'), index=False, encoding='utf-8-sig')
    print(f"\n>> 저장: bi3_L1b_chronic_aoi.csv")

    # AOI Concept Description — 각 질환 그룹의 주요 특징
    print("\n[AOI Concept Description: 질환별 특징값]")
    for dis_val in df_chronic[dis_col].dropna().unique():
        sub = df_chronic[df_chronic[dis_col] == dis_val]
        desc = {}
        for nc in num_chr:
            vals = sub[nc].dropna()
            if len(vals) > 0:
                desc[nc] = {'평균': round(vals.mean(), 2), '최대': round(vals.max(), 2)}
        print(f"  [{dis_val}] {desc}")

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 : Subspace Clustering (KMeans, 2개 서브스페이스)
# ══════════════════════════════════════════════════════════════════════════════
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

header("LAYER 2-A : Subspace Clustering — CAT 서브스페이스 (서비스구분×등록유형×성별)")

# ── 범주형 컬럼 확인
cat_cols_candidates = ['서비스구분', '등록유형', '성별', '지역']
exist_cat = [c for c in cat_cols_candidates if c in df_quit.columns]
print(f"존재하는 CAT 컬럼: {exist_cat}")
for c in exist_cat:
    vc = df_quit[c].value_counts()
    print(f"  {c}: {vc.to_dict()}")

# ── 샘플링 (239K 행 → 30K)
np.random.seed(42)
sample_idx = np.random.choice(len(df_quit), size=min(30000, len(df_quit)), replace=False)
df_sample = df_quit.iloc[sample_idx].copy().reset_index(drop=True)

# Label Encoding
le_dict = {}
X_cat = pd.DataFrame()
for c in exist_cat:
    le = LabelEncoder()
    col_clean = df_sample[c].astype(str).str.strip().fillna('미상')
    X_cat[c + '_enc'] = le.fit_transform(col_clean)
    le_dict[c] = dict(zip(le.classes_, le.transform(le.classes_)))

print(f"\nCAT 서브스페이스 shape: {X_cat.shape}")

# Elbow 방법으로 최적 k 결정
inertias_cat = []
sil_cat = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_cat)
    inertias_cat.append(km.inertia_)
    if k >= 2:
        sil_cat.append(silhouette_score(X_cat, labels, sample_size=5000))

best_k_cat = K_range[np.argmax(sil_cat)]
print(f"\nCAT 실루엣 점수: {[round(s,3) for s in sil_cat]}")
print(f"최적 k (CAT) = {best_k_cat}")

km_cat = KMeans(n_clusters=best_k_cat, random_state=42, n_init=10)
df_sample['cluster_cat'] = km_cat.fit_predict(X_cat)

# 클러스터 프로파일
print("\n[CAT 클러스터 프로파일]")
cat_profile = df_sample.groupby('cluster_cat')[exist_cat].agg(lambda x: x.value_counts().index[0]).reset_index()
cat_counts = df_sample['cluster_cat'].value_counts().reset_index()
cat_counts.columns = ['cluster_cat', 'n']
cat_profile = cat_profile.merge(cat_counts, on='cluster_cat')
print(cat_profile.to_string(index=False))
cat_profile.to_csv(os.path.join(OUT, 'bi3_L2a_cat_cluster_profile.csv'), index=False, encoding='utf-8-sig')

header("LAYER 2-B : Subspace Clustering — NUM 서브스페이스 (상담횟수×CO측정값)")

# ── 수치형 컬럼 확인
num_candidates = ['상담횟수']
# CO 측정값 컬럼 탐색
for c in df_quit.columns:
    if 'co' in c.lower() or 'CO' in c or '코티닌' in c or '일산화탄소' in c:
        num_candidates.append(c)
# 추가 수치 컬럼
for c in df_quit.columns:
    if '횟수' in c or '측정' in c or '값' in c:
        if c not in num_candidates:
            num_candidates.append(c)

exist_num = [c for c in num_candidates if c in df_quit.columns]
print(f"존재하는 NUM 컬럼 후보: {exist_num}")
for c in exist_num:
    ser = pd.to_numeric(df_quit[c], errors='coerce')
    print(f"  {c}: non-null={ser.notna().sum()}, mean={ser.mean():.2f}, max={ser.max():.1f}")

# non-null 비율 50% 이상인 컬럼만 사용
valid_num = []
for c in exist_num:
    ser = pd.to_numeric(df_quit[c], errors='coerce')
    if ser.notna().sum() / len(df_quit) >= 0.3:
        valid_num.append(c)
print(f"\n사용 NUM 컬럼 (non-null ≥ 30%): {valid_num}")

if len(valid_num) >= 2:
    X_num_full = pd.DataFrame()
    for c in valid_num:
        X_num_full[c] = pd.to_numeric(df_sample[c], errors='coerce').fillna(0)
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num_full)

    sil_num = []
    for k in K_range:
        km_n = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_n = km_n.fit_predict(X_num_scaled)
        sil_num.append(silhouette_score(X_num_scaled, labels_n, sample_size=5000))

    best_k_num = K_range[np.argmax(sil_num)]
    print(f"\nNUM 실루엣 점수: {[round(s,3) for s in sil_num]}")
    print(f"최적 k (NUM) = {best_k_num}")

    km_num = KMeans(n_clusters=best_k_num, random_state=42, n_init=10)
    df_sample['cluster_num'] = km_num.fit_predict(X_num_scaled)

    print("\n[NUM 클러스터 프로파일 (평균)]")
    num_profile = df_sample.groupby('cluster_num')[valid_num].apply(
        lambda x: x.apply(pd.to_numeric, errors='coerce').mean()
    ).round(2).reset_index()
    num_counts = df_sample['cluster_num'].value_counts().reset_index()
    num_counts.columns = ['cluster_num', 'n']
    num_profile = num_profile.merge(num_counts, on='cluster_num')
    print(num_profile.to_string(index=False))
    num_profile.to_csv(os.path.join(OUT, 'bi3_L2b_num_cluster_profile.csv'), index=False, encoding='utf-8-sig')

    # ── COMBINED 클러스터 교차표
    if 'cluster_cat' in df_sample.columns:
        cross = pd.crosstab(df_sample['cluster_cat'], df_sample['cluster_num'],
                            margins=True, margins_name='합계')
        print("\n[CAT × NUM 클러스터 교차표]")
        print(cross)
        cross.to_csv(os.path.join(OUT, 'bi3_L2c_cluster_crosstab.csv'), encoding='utf-8-sig')

        # 각 CAT 클러스터에서 성공률 계산
        eval_cols_check = ['4주평가성공여부', '6주평가성공여부', '12주평가성공여부', '6개월평가성공여부']
        exist_eval = [c for c in eval_cols_check if c in df_sample.columns]
        if exist_eval:
            for ec in exist_eval:
                df_sample[ec+'_Y'] = (df_sample[ec].astype(str).str.strip() == 'Y').astype(int)
            y_cols = [c+'_Y' for c in exist_eval]
            cluster_success = df_sample.groupby('cluster_cat')[y_cols].mean().round(3)
            print("\n[CAT 클러스터별 각 단계 성공률]")
            print(cluster_success)
            cluster_success.to_csv(os.path.join(OUT, 'bi3_L2d_cluster_success_rate.csv'), encoding='utf-8-sig')
else:
    print("  수치형 컬럼 부족 → NUM 클러스터링 생략, cluster_num 더미 설정")
    df_sample['cluster_num'] = 0

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 : FP-Growth + Association Rules (2개 계층)
# ══════════════════════════════════════════════════════════════════════════════
from mlxtend.frequent_patterns import fpgrowth, association_rules

header("LAYER 3-A : FP-Growth — 등록정보 서비스구분×등록유형×서비스상태 패턴")

# 트랜잭션 구성: 각 행을 서비스구분, 등록유형, 서비스상태, 성별의 값 조합으로
item_cols = []
for c in ['서비스구분', '등록유형', '서비스상태', '성별']:
    if c in df_quit.columns:
        item_cols.append(c)
print(f"트랜잭션 구성 컬럼: {item_cols}")

# 성공 평가 여부 추가
eval_item_cols = [c for c in ['4주평가성공여부', '6주평가성공여부', '12주평가성공여부', '6개월평가성공여부']
                 if c in df_quit.columns]
item_cols.extend(eval_item_cols)

# 샘플 사용 (50K)
np.random.seed(42)
n_fp = min(50000, len(df_quit))
df_fp = df_quit.sample(n=n_fp, random_state=42).copy()

# 아이템 생성: "컬럼명=값" 형식
all_items = set()
def row_to_items(row, cols):
    items = []
    for c in cols:
        val = str(row[c]).strip() if pd.notna(row[c]) else None
        if val and val not in ['nan', 'None', '']:
            items.append(f"{c}={val}")
    return items

transactions = df_fp.apply(lambda r: row_to_items(r, item_cols), axis=1).tolist()
transactions = [t for t in transactions if len(t) >= 2]
print(f"트랜잭션 수: {len(transactions)}")

# 아이템 전체 목록
all_items = sorted(set(item for t in transactions for item in t))
print(f"아이템 수: {len(all_items)}")

# 원핫 인코딩
item_to_idx = {item: i for i, item in enumerate(all_items)}
n_trans = len(transactions)
n_items = len(all_items)

# sparse 방식으로 변환
from scipy.sparse import lil_matrix
mat = lil_matrix((n_trans, n_items), dtype=bool)
for i, trans in enumerate(transactions):
    for item in trans:
        if item in item_to_idx:
            mat[i, item_to_idx[item]] = True

df_onehot = pd.DataFrame.sparse.from_spmatrix(mat, columns=all_items)

# FP-Growth
print("\nFP-Growth 실행 중 (min_support=0.05)...")
try:
    freq_items = fpgrowth(df_onehot, min_support=0.05, use_colnames=True, max_len=4)
    print(f"빈발 아이템셋 수: {len(freq_items)}")
    
    if len(freq_items) > 0:
        rules_global = association_rules(freq_items, metric='lift', min_threshold=1.2)
        rules_global = rules_global.sort_values('lift', ascending=False)
        print(f"연관 규칙 수: {len(rules_global)}")
        
        # antecedents/consequents를 문자열로 변환
        rules_global['antecedents_str'] = rules_global['antecedents'].apply(lambda x: ', '.join(sorted(x)))
        rules_global['consequents_str'] = rules_global['consequents'].apply(lambda x: ', '.join(sorted(x)))
        
        out_cols = ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift', 'leverage']
        print("\n[Top 20 연관 규칙 (lift 기준)]")
        print(rules_global[out_cols].head(20).to_string(index=False))
        
        rules_save = rules_global[out_cols].copy()
        rules_save.to_csv(os.path.join(OUT, 'bi3_L3a_assoc_rules_global.csv'), index=False, encoding='utf-8-sig')
        print(f"\n>> 저장: bi3_L3a_assoc_rules_global.csv ({len(rules_save)}개 규칙)")
    else:
        print("  빈발 아이템셋 없음 (support 너무 낮음) → min_support 낮춰서 재시도")
        rules_global = pd.DataFrame()
except Exception as e:
    print(f"  FP-Growth 오류: {e}")
    rules_global = pd.DataFrame()

header("LAYER 3-B : FP-Growth — 물품구매 동반 패턴 (치료 프로토콜 단계)")
print("[물품구매 컬럼]", list(df_goods.columns))
print(df_goods.head(10).to_string())

# 구분 컬럼의 값: 껌2mg, 패치1단계, 패치2단계, 패치3단계
goods_item_col = None
for c in df_goods.columns:
    if '구분' in c or '품목' in c or '물품' in c:
        goods_item_col = c
        break
region_col_goods = None
for c in df_goods.columns:
    if '지역' in c:
        region_col_goods = c
        break

print(f"\n물품 아이템 컬럼: '{goods_item_col}', 지역 컬럼: '{region_col_goods}'")
if goods_item_col:
    print(df_goods[goods_item_col].value_counts())

# 지역별 구매 물품 트랜잭션 (지역 = 트랜잭션 단위)
if goods_item_col and region_col_goods:
    goods_trans = df_goods.groupby(region_col_goods)[goods_item_col].apply(
        lambda x: list(x.dropna().unique())
    ).tolist()
    goods_trans = [t for t in goods_trans if len(t) >= 1]
    print(f"\n지역별 트랜잭션 수: {len(goods_trans)}")
    
    all_goods = sorted(set(item for t in goods_trans for item in t))
    print(f"물품 종류: {all_goods}")
    
    if len(goods_trans) >= 5 and len(all_goods) >= 2:
        g_dict = {item: i for i, item in enumerate(all_goods)}
        g_mat = pd.DataFrame(False, index=range(len(goods_trans)), columns=all_goods)
        for i, trans in enumerate(goods_trans):
            for item in trans:
                if item in g_dict:
                    g_mat.loc[i, item] = True
        
        try:
            g_freq = fpgrowth(g_mat, min_support=0.3, use_colnames=True)
            print(f"물품 빈발셋: {len(g_freq)}")
            if len(g_freq) > 0:
                g_rules = association_rules(g_freq, metric='lift', min_threshold=1.0)
                g_rules['antecedents_str'] = g_rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
                g_rules['consequents_str'] = g_rules['consequents'].apply(lambda x: ', '.join(sorted(x)))
                out_g = ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']
                print("\n[물품 연관 규칙]")
                print(g_rules[out_g].sort_values('lift', ascending=False).head(20).to_string(index=False))
                g_rules[out_g].sort_values('lift', ascending=False).to_csv(
                    os.path.join(OUT, 'bi3_L3b_goods_assoc_rules.csv'), index=False, encoding='utf-8-sig')
                print(f"\n>> 저장: bi3_L3b_goods_assoc_rules.csv")
        except Exception as e:
            print(f"  물품 FP-Growth 오류: {e}")

# 물품 구분별 구매량 통계 (치료 단계 순서)
if goods_item_col:
    qty_col = None
    for c in df_goods.columns:
        if '수량' in c or '구매' in c and c != goods_item_col:
            qty_col = c
            break
    if qty_col:
        stage_summary = df_goods.groupby(goods_item_col)[qty_col].agg(['sum','mean','count']).reset_index()
        stage_summary.columns = [goods_item_col, '총구매량', '평균구매량', '구매건수']
        # 치료 단계 순서 정렬
        stage_order = {'껌2mg': 1, '패치1단계': 2, '패치2단계': 3, '패치3단계': 4}
        stage_summary['순서'] = stage_summary[goods_item_col].map(stage_order).fillna(99)
        stage_summary = stage_summary.sort_values('순서')
        print("\n[치료 프로토콜 단계별 물품 구매량]")
        print(stage_summary.to_string(index=False))
        stage_summary.to_csv(os.path.join(OUT, 'bi3_L3c_treatment_stage_summary.csv'), index=False, encoding='utf-8-sig')
        print(f"\n>> 저장: bi3_L3c_treatment_stage_summary.csv")

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 : Sequential Pattern Mining
# ══════════════════════════════════════════════════════════════════════════════

header("LAYER 4-A : Sequential Patterns — 금연 4단계 성공/실패 시퀀스")

# 4단계 평가 성공여부 (Y / 미실시 / null → 드롭아웃)
eval_cols = ['4주평가성공여부', '6주평가성공여부', '12주평가성공여부', '6개월평가성공여부']
exist_eval = [c for c in eval_cols if c in df_quit.columns]
print(f"존재하는 평가 컬럼: {exist_eval}")

if exist_eval:
    # 각 레코드를 시퀀스로 변환
    def encode_stage(val):
        v = str(val).strip() if pd.notna(val) else 'DROP'
        if v == 'Y':
            return 'Y'
        elif v in ['미실시', 'N', '']:
            return 'N'
        elif v == 'DROP' or v == 'nan' or v == 'None':
            return 'DROP'
        else:
            return 'N'

    seq_df = df_quit[exist_eval].copy()
    for c in exist_eval:
        seq_df[c + '_seq'] = seq_df[c].apply(encode_stage)

    seq_cols = [c + '_seq' for c in exist_eval]
    seq_df['sequence'] = seq_df[seq_cols].apply(lambda r: '→'.join(r.values), axis=1)
    seq_df['n_stages'] = len(exist_eval)

    # 시퀀스 빈도 집계
    seq_counts = seq_df['sequence'].value_counts().reset_index()
    seq_counts.columns = ['sequence', 'count']
    seq_counts['ratio'] = (seq_counts['count'] / len(seq_df)).round(4)
    print(f"\n전체 레코드: {len(seq_df):,}")
    print(f"고유 시퀀스 수: {len(seq_counts)}")
    print("\n[Top 30 금연 시퀀스 패턴]")
    print(seq_counts.head(30).to_string(index=False))
    seq_counts.to_csv(os.path.join(OUT, 'bi3_L4a_quit_sequences.csv'), index=False, encoding='utf-8-sig')
    print(f"\n>> 저장: bi3_L4a_quit_sequences.csv")

    # 완전 성공 vs 조기 종결 분석
    all_Y = seq_df['sequence'].str.count('Y')
    print(f"\n[성공 단계 분포]")
    stage_dist = all_Y.value_counts().sort_index().reset_index()
    stage_dist.columns = ['성공단계수', '인원수']
    stage_dist['비율'] = (stage_dist['인원수'] / len(seq_df)).round(4)
    print(stage_dist.to_string(index=False))
    stage_dist.to_csv(os.path.join(OUT, 'bi3_L4a_success_stage_dist.csv'), index=False, encoding='utf-8-sig')

    # 서비스구분별 시퀀스 패턴
    if '서비스구분' in df_quit.columns:
        seq_df2 = seq_df.copy()
        seq_df2['서비스구분'] = df_quit['서비스구분'].values
        svc_seq = seq_df2.groupby(['서비스구분', 'sequence']).size().reset_index(name='count')
        svc_total = svc_seq.groupby('서비스구분')['count'].sum().reset_index(name='total')
        svc_seq = svc_seq.merge(svc_total, on='서비스구분')
        svc_seq['ratio'] = (svc_seq['count'] / svc_seq['total']).round(4)
        svc_seq = svc_seq.sort_values(['서비스구분', 'count'], ascending=[True, False])
        print("\n[서비스구분별 Top 5 시퀀스]")
        print(svc_seq.groupby('서비스구분').head(5).to_string(index=False))
        svc_seq.to_csv(os.path.join(OUT, 'bi3_L4a_svc_seq_patterns.csv'), index=False, encoding='utf-8-sig')
        print(f"\n>> 저장: bi3_L4a_svc_seq_patterns.csv")

header("LAYER 4-B : Sequential Patterns — 방문연계기관 시퀀스 (연계기관유형 방문 순서)")
print("[방문연계 컬럼]")
for c in df_visit.columns:
    print(f"  '{c}': {df_visit[c].dtype} | unique={df_visit[c].nunique()} | sample: {list(df_visit[c].dropna().unique()[:3])}")

# 연계일자 기반 순서 분석
date_col_v = None
for c in df_visit.columns:
    if '일자' in c or '날짜' in c or '일시' in c:
        date_col_v = c
        break
type_col_v = None
for c in df_visit.columns:
    if '유형' in c or '종류' in c:
        type_col_v = c
        break
region_col_v = None
for c in df_visit.columns:
    if '지역' in c:
        region_col_v = c
        break
org_col_v = None
for c in df_visit.columns:
    if '기관명' in c and '연계' not in c:
        org_col_v = c
        break

print(f"\n날짜='{date_col_v}', 연계유형='{type_col_v}', 지역='{region_col_v}', 기관명='{org_col_v}'")

if date_col_v and type_col_v:
    df_visit2 = df_visit.copy()
    df_visit2[date_col_v] = pd.to_datetime(df_visit2[date_col_v], errors='coerce')
    df_visit2 = df_visit2.dropna(subset=[date_col_v])
    df_visit2['yearmonth'] = df_visit2[date_col_v].dt.to_period('M')
    
    # 월별 연계유형 시퀀스
    monthly_type = df_visit2.groupby(['yearmonth', type_col_v]).size().reset_index(name='count')
    monthly_type = monthly_type.sort_values(['yearmonth', 'count'], ascending=[True, False])
    print("\n[월별 연계기관유형 분포 (상위)]")
    print(monthly_type.head(20).to_string(index=False))
    monthly_type['yearmonth'] = monthly_type['yearmonth'].astype(str)
    monthly_type.to_csv(os.path.join(OUT, 'bi3_L4b_visit_monthly_type.csv'), index=False, encoding='utf-8-sig')
    print(f"\n>> 저장: bi3_L4b_visit_monthly_type.csv")

    # 연계유형 전이 행렬 (순차적 방문 패턴)
    df_visit_sorted = df_visit2.sort_values(date_col_v)
    types_seq = df_visit_sorted[type_col_v].dropna().tolist()
    
    transition_count = defaultdict(int)
    for i in range(len(types_seq) - 1):
        a, b = str(types_seq[i]).strip(), str(types_seq[i+1]).strip()
        if a and b and a != 'nan' and b != 'nan':
            transition_count[(a, b)] += 1
    
    trans_df = pd.DataFrame([(a, b, c) for (a, b), c in transition_count.items()],
                             columns=['from_type', 'to_type', 'count'])
    trans_df = trans_df.sort_values('count', ascending=False)
    print("\n[연계기관유형 전이 패턴 Top 20]")
    print(trans_df.head(20).to_string(index=False))
    trans_df.to_csv(os.path.join(OUT, 'bi3_L4b_visit_transition.csv'), index=False, encoding='utf-8-sig')
    print(f"\n>> 저장: bi3_L4b_visit_transition.csv")

    # 전이 행렬 피벗
    if len(trans_df) > 0:
        trans_pivot = trans_df.pivot(index='from_type', columns='to_type', values='count').fillna(0)
        print("\n[연계기관유형 전이 행렬]")
        print(trans_pivot)
        trans_pivot.to_csv(os.path.join(OUT, 'bi3_L4b_visit_transition_matrix.csv'), encoding='utf-8-sig')

header("LAYER 4-C : Sequential Patterns — 심박수 상태 시계열 전이")
print("[심박수 컬럼]")
for c in df_heart.columns:
    print(f"  '{c}': {df_heart[c].dtype} | sample: {list(df_heart[c].dropna().unique()[:3])}")

heart_date_col = None
heart_val_col = None
for c in df_heart.columns:
    if '일시' in c or '날짜' in c or '일자' in c:
        heart_date_col = c
    if '심박측정' == c or ('심박' in c and '측정' in c and '최고' not in c and '최저' not in c and '배열' not in c):
        heart_val_col = c

print(f"\n날짜컬럼='{heart_date_col}', 심박값컬럼='{heart_val_col}'")

if heart_date_col and heart_val_col:
    df_heart2 = df_heart.copy()
    df_heart2[heart_date_col] = pd.to_datetime(df_heart2[heart_date_col], errors='coerce')
    df_heart2[heart_val_col] = pd.to_numeric(df_heart2[heart_val_col], errors='coerce')
    df_heart2 = df_heart2.dropna(subset=[heart_date_col, heart_val_col])
    
    # 심박수 구간화: 정상(60-100bpm), 서맥(<60), 빈맥(>100)
    def classify_hr(bpm):
        if bpm < 60:
            return '서맥(저박동)'
        elif bpm <= 100:
            return '정상'
        else:
            return '빈맥(고박동)'
    
    df_heart2['hr_state'] = df_heart2[heart_val_col].apply(classify_hr)
    df_heart2 = df_heart2.sort_values(heart_date_col)
    
    print(f"\n심박 상태 분포:")
    print(df_heart2['hr_state'].value_counts())
    
    # 상태 전이 분석
    states = df_heart2['hr_state'].tolist()
    hr_transition = defaultdict(int)
    for i in range(len(states) - 1):
        hr_transition[(states[i], states[i+1])] += 1
    
    hr_trans_df = pd.DataFrame([(a, b, c) for (a, b), c in hr_transition.items()],
                                columns=['from_state', 'to_state', 'count'])
    hr_trans_df = hr_trans_df.sort_values('count', ascending=False)
    
    total_trans = hr_trans_df['count'].sum()
    hr_trans_df['prob'] = (hr_trans_df['count'] / total_trans).round(4)
    print("\n[심박수 상태 전이 패턴]")
    print(hr_trans_df.to_string(index=False))
    hr_trans_df.to_csv(os.path.join(OUT, 'bi3_L4c_heart_transition.csv'), index=False, encoding='utf-8-sig')
    print(f"\n>> 저장: bi3_L4c_heart_transition.csv")

    # 월별 심박 상태 분포
    df_heart2['yearmonth'] = df_heart2[heart_date_col].dt.to_period('M')
    hr_monthly = df_heart2.groupby(['yearmonth', 'hr_state']).size().reset_index(name='count')
    hr_monthly['yearmonth'] = hr_monthly['yearmonth'].astype(str)
    hr_monthly.to_csv(os.path.join(OUT, 'bi3_L4c_heart_monthly.csv'), index=False, encoding='utf-8-sig')

    # 전이 확률 행렬
    hr_pivot = hr_trans_df.pivot(index='from_state', columns='to_state', values='prob').fillna(0)
    # 행 정규화
    hr_pivot_norm = hr_pivot.div(hr_pivot.sum(axis=1), axis=0).round(4)
    print("\n[심박 상태 전이 확률 행렬 (행 정규화)]")
    print(hr_pivot_norm)
    hr_pivot_norm.to_csv(os.path.join(OUT, 'bi3_L4c_heart_transition_matrix.csv'), encoding='utf-8-sig')

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY : 종합 인사이트 JSON
# ══════════════════════════════════════════════════════════════════════════════
header("SUMMARY : 건강증진 BI-3 분석 결과 요약")

summary = {
    "analysis_name": "BI-3 건강증진 프로그램 참여 순서 및 효과 Sequential 패턴 탐사",
    "layers": {
        "L1_AOI": {
            "desc": "AOI Concept Description - 정책 목표 달성도 및 만성질환 참여강도 비교",
            "outputs": ["bi3_L1a_policy_aoi.csv", "bi3_L1a_achieve_stats.csv", "bi3_L1b_chronic_aoi.csv"],
            "key_findings": []
        },
        "L2_Subspace_Clustering": {
            "desc": "KMeans Subspace Clustering - CAT/NUM 2개 서브스페이스",
            "outputs": ["bi3_L2a_cat_cluster_profile.csv", "bi3_L2b_num_cluster_profile.csv",
                        "bi3_L2c_cluster_crosstab.csv", "bi3_L2d_cluster_success_rate.csv"],
            "key_findings": []
        },
        "L3_FPGrowth": {
            "desc": "FP-Growth + Association Rules - 서비스 조합 패턴 및 물품 구매 패턴",
            "outputs": ["bi3_L3a_assoc_rules_global.csv", "bi3_L3b_goods_assoc_rules.csv",
                        "bi3_L3c_treatment_stage_summary.csv"],
            "key_findings": []
        },
        "L4_Sequential": {
            "desc": "Sequential Pattern Analysis - 금연 4단계/방문연계/심박수 상태 시퀀스",
            "outputs": ["bi3_L4a_quit_sequences.csv", "bi3_L4a_success_stage_dist.csv",
                        "bi3_L4a_svc_seq_patterns.csv", "bi3_L4b_visit_monthly_type.csv",
                        "bi3_L4b_visit_transition.csv", "bi3_L4b_visit_transition_matrix.csv",
                        "bi3_L4c_heart_transition.csv", "bi3_L4c_heart_monthly.csv",
                        "bi3_L4c_heart_transition_matrix.csv"],
            "key_findings": []
        }
    },
    "data_sources": {
        "금연서비스등록정보": f"{len(df_quit):,}행 × {df_quit.shape[1]}컬럼",
        "심박수": f"{len(df_heart):,}행 × {df_heart.shape[1]}컬럼",
        "방문연계기관": f"{len(df_visit):,}행 × {df_visit.shape[1]}컬럼",
        "물품구매현황": f"{len(df_goods):,}행 × {df_goods.shape[1]}컬럼",
        "건강증진종합계획": f"{len(df_policy):,}행 × {df_policy.shape[1]}컬럼",
        "만성질환관리통계": f"{len(df_chronic):,}행 × {df_chronic.shape[1]}컬럼"
    }
}

# 동적 인사이트 추가
try:
    # L1 인사이트
    if os.path.exists(os.path.join(OUT, 'bi3_L1a_achieve_stats.csv')):
        df_ach = pd.read_csv(os.path.join(OUT, 'bi3_L1a_achieve_stats.csv'), encoding='utf-8-sig')
        if len(df_ach) > 0:
            best = df_ach.iloc[0]
            worst = df_ach.iloc[-1]
            summary["layers"]["L1_AOI"]["key_findings"].append(
                f"최고 달성 과제: {best.iloc[0]} (평균 {best['mean_achievement']:.1f}%)"
            )
            summary["layers"]["L1_AOI"]["key_findings"].append(
                f"최저 달성 과제: {worst.iloc[0]} (평균 {worst['mean_achievement']:.1f}%)"
            )
except: pass

try:
    # L4 인사이트
    if os.path.exists(os.path.join(OUT, 'bi3_L4a_quit_sequences.csv')):
        df_seq = pd.read_csv(os.path.join(OUT, 'bi3_L4a_quit_sequences.csv'), encoding='utf-8-sig')
        if len(df_seq) > 0:
            top_seq = df_seq.iloc[0]
            summary["layers"]["L4_Sequential"]["key_findings"].append(
                f"최빈 시퀀스: {top_seq['sequence']} (비율: {top_seq['ratio']:.1%})"
            )
    if os.path.exists(os.path.join(OUT, 'bi3_L4a_success_stage_dist.csv')):
        df_sd = pd.read_csv(os.path.join(OUT, 'bi3_L4a_success_stage_dist.csv'), encoding='utf-8-sig')
        all_Y_row = df_sd[df_sd['성공단계수'] == len(exist_eval)] if len(exist_eval) > 0 else pd.DataFrame()
        if len(all_Y_row) > 0:
            summary["layers"]["L4_Sequential"]["key_findings"].append(
                f"전 단계 성공(완전금연) 비율: {all_Y_row.iloc[0]['비율']:.1%}"
            )
except: pass

with open(os.path.join(OUT, 'bi3_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(">> 저장: bi3_summary.json")

# 출력 파일 목록
print(f"\n[{OUT}] 생성된 파일:")
for fname in sorted(os.listdir(OUT)):
    fpath = os.path.join(OUT, fname)
    fsize = os.path.getsize(fpath)
    print(f"  {fname:55s}  {fsize:>10,} bytes")

print(f"\n{'='*72}")
print("  BI-3 건강증진 Sequential 패턴 분석 완료!")
print(f"{'='*72}")
