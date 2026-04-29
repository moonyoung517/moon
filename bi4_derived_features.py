# -*- coding: utf-8 -*-
# bi4_derived_features.py
# 신규 파생 컬럼 + 범주화 기반 추가 분석
# - 상담강도_구간 (pd.cut)
# - 금연달성등급 (총성공단계 → 4레벨)
# - 지역권역 (수도권/비수도권)
# - 초기CO위험도 (4주 CO값 범주화)
# - 출생코호트 (출생년도 → 세대 구분)
# - 상담강도별 × 달성등급 교차분석
# - 파생 피처 추가 KMeans 재클러스터링
import sys, os, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RAW = r'd:\GIT\others\data_intelligence\bi_rawdata'
OUT = r'd:\GIT\others\data_intelligence\governance_bi\bi3_output'

F_QUIT = os.path.join(RAW, '한국건강증진개발원_국가금연지원서비스 등록정보(기본)_20241231.csv')

sep = "=" * 72
def header(t): print(f"\n{sep}\n  {t}\n{sep}")

# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────
header("데이터 로드")
df = pd.read_csv(F_QUIT, encoding='cp949', low_memory=False)
df.columns = df.columns.str.strip()
print(f"원본 데이터: {df.shape[0]:,}행 × {df.shape[1]}컬럼")
print(f"컬럼 목록: {list(df.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: 파생 컬럼 생성
# ─────────────────────────────────────────────────────────────────────────────
header("STEP 1: 파생 컬럼 생성")

# 1-1. 성공여부 이진 인코딩 (_bin)
eval_cols = ['4주평가성공여부', '6주평가성공여부', '12주평가성공여부', '6개월평가성공여부']
for c in eval_cols:
    df[c + '_bin'] = (df[c].astype(str).str.strip() == 'Y').astype(int)
print(f"파생: _bin 컬럼 4개 생성")

# 1-2. 총성공단계 (0~4)
df['총성공단계'] = df[[c + '_bin' for c in eval_cols]].sum(axis=1)
print(f"파생: 총성공단계 (0~4)  분포:\n{df['총성공단계'].value_counts().sort_index()}")

# 1-3. 상담횟수 수치화
df['상담횟수_num'] = pd.to_numeric(df['상담횟수'], errors='coerce').fillna(0)

# 1-4. 상담강도_구간 (pd.cut — 수치 → 4단계 범주)
bins_consult = [0, 5, 10, 15, df['상담횟수_num'].max() + 1]
labels_consult = ['저강도(1-5회)', '중강도(6-10회)', '고강도(11-15회)', '집중(16회+)']
df['상담강도_구간'] = pd.cut(df['상담횟수_num'], bins=bins_consult, labels=labels_consult, right=True)
df['상담강도_구간'] = df['상담강도_구간'].astype(str)
print(f"\n파생: 상담강도_구간 분포:\n{df['상담강도_구간'].value_counts()}")

# 1-5. 금연달성등급 (총성공단계 → 4레벨 범주)
bins_success = [-1, 0, 1, 3, 4]
labels_success = ['미달성(0단계)', '초기달성(1단계)', '부분달성(2-3단계)', '완전달성(4단계)']
df['금연달성등급'] = pd.cut(df['총성공단계'], bins=bins_success, labels=labels_success, right=True)
df['금연달성등급'] = df['금연달성등급'].astype(str)
print(f"\n파생: 금연달성등급 분포:\n{df['금연달성등급'].value_counts()}")

# 1-6. 지역권역 (수도권/비수도권)
수도권 = ['서울', '경기도', '인천']
df['지역_cleaned'] = df['지역'].astype(str).str.strip()
df['지역권역'] = df['지역_cleaned'].apply(lambda x: '수도권' if x in 수도권 else '비수도권')
print(f"\n파생: 지역권역 분포:\n{df['지역권역'].value_counts()}")

# 1-7. 초기CO위험도 (4주 CO 측정값 → 3단계 범주)
co_col = '4주평가_측정값(일산화탄소)'
df[co_col + '_num'] = pd.to_numeric(df[co_col], errors='coerce')
co_measured = df[co_col + '_num'].dropna()
print(f"\n4주 CO 측정값 통계: {co_measured.describe().round(2)}")

def classify_co(val):
    if pd.isna(val):
        return '미측정'
    elif val <= 5:
        return '정상(0-5ppm)'
    elif val <= 15:
        return '경계(6-15ppm)'
    else:
        return '위험(16ppm+)'

df['초기CO위험도'] = df[co_col + '_num'].apply(classify_co)
print(f"\n파생: 초기CO위험도 분포:\n{df['초기CO위험도'].value_counts()}")

# 1-8. 출생코호트 (출생년도 구간 → 세대 범주)
birth_col = '출생년도'
if birth_col in df.columns:
    df[birth_col + '_str'] = df[birth_col].astype(str).str.strip()
    def classify_cohort(val):
        if '1950' in val or '1960' in val:
            return '시니어(1950-1969)'
        elif '1970' in val:
            return '50대(1970년대)'
        elif '1980' in val:
            return '40대(1980년대)'
        elif '1990' in val:
            return '30대(1990년대)'
        elif '2000' in val or '2010' in val:
            return '20대이하(2000년대+)'
        else:
            return '기타'
    df['출생코호트'] = df[birth_col + '_str'].apply(classify_cohort)
    print(f"\n파생: 출생코호트 분포:\n{df['출생코호트'].value_counts()}")

print(f"\n총 생성된 파생 컬럼: 총성공단계, 상담강도_구간, 금연달성등급, 지역권역, 초기CO위험도, 출생코호트")
print(f"현재 데이터 컬럼 수: {df.shape[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: 교차 분석 — 상담강도_구간 × 금연달성등급
# ─────────────────────────────────────────────────────────────────────────────
header("STEP 2: 상담강도_구간 × 금연달성등급 교차분석")

cross1 = pd.crosstab(df['상담강도_구간'], df['금연달성등급'], margins=False)
cross1_pct = cross1.div(cross1.sum(axis=1), axis=0).round(4) * 100
print("\n[절대 인원수]")
print(cross1.to_string())
print("\n[비율 (행 기준, %)]")
print(cross1_pct.to_string())

# 상담강도별 완전달성 비율
완전달성률 = cross1_pct['완전달성(4단계)'] if '완전달성(4단계)' in cross1_pct.columns else None
if 완전달성률 is not None:
    print(f"\n▶ 상담강도별 완전달성(4단계) 비율:")
    print(완전달성률.sort_values(ascending=False).to_string())

cross1.to_csv(os.path.join(OUT, 'bi4_derived_intensity_grade_cross.csv'), encoding='utf-8-sig')
cross1_pct.to_csv(os.path.join(OUT, 'bi4_derived_intensity_grade_pct.csv'), encoding='utf-8-sig')
print(f"\n>> 저장: bi4_derived_intensity_grade_cross.csv, bi4_derived_intensity_grade_pct.csv")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: 교차 분석 — 지역권역 × 달성등급 × 서비스구분
# ─────────────────────────────────────────────────────────────────────────────
header("STEP 3: 지역권역 × 금연달성등급 × 서비스구분 교차분석")

cross2 = pd.crosstab([df['지역권역'], df['서비스구분']], df['금연달성등급'])
cross2_pct = cross2.div(cross2.sum(axis=1), axis=0).round(4) * 100
print("[지역권역 × 서비스구분별 달성등급 분포 (비율%)]")
print(cross2_pct.to_string())

cross2_pct.to_csv(os.path.join(OUT, 'bi4_derived_region_svc_grade.csv'), encoding='utf-8-sig')
print(f"\n>> 저장: bi4_derived_region_svc_grade.csv")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: 초기CO위험도 × 금연달성등급 교차분석
# ─────────────────────────────────────────────────────────────────────────────
header("STEP 4: 초기CO위험도 × 금연달성등급 교차분析")

cross3 = pd.crosstab(df['초기CO위험도'], df['금연달성등급'])
cross3_pct = cross3.div(cross3.sum(axis=1), axis=0).round(4) * 100
print("[초기CO위험도별 달성등급 분포 (비율%)]")
print(cross3_pct.to_string())

cross3.to_csv(os.path.join(OUT, 'bi4_derived_co_grade_cross.csv'), encoding='utf-8-sig')
cross3_pct.to_csv(os.path.join(OUT, 'bi4_derived_co_grade_pct.csv'), encoding='utf-8-sig')
print(f"\n>> 저장: bi4_derived_co_grade_cross.csv, bi4_derived_co_grade_pct.csv")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: 출생코호트 × 상담강도_구간 × 금연달성등급
# ─────────────────────────────────────────────────────────────────────────────
if '출생코호트' in df.columns:
    header("STEP 5: 출생코호트 × 상담강도_구간 × 완전달성 비율")
    cohort_success = df.groupby(['출생코호트', '상담강도_구간']).agg(
        인원수=('총성공단계', 'count'),
        평균성공단계=('총성공단계', 'mean'),
        완전달성수=('총성공단계', lambda x: (x == 4).sum())
    ).reset_index()
    cohort_success['완전달성률(%)'] = (cohort_success['완전달성수'] / cohort_success['인원수'] * 100).round(2)
    cohort_success = cohort_success.sort_values(['출생코호트', '완전달성률(%)'], ascending=[True, False])
    print(cohort_success.to_string(index=False))
    cohort_success.to_csv(os.path.join(OUT, 'bi4_derived_cohort_intensity.csv'), index=False, encoding='utf-8-sig')
    print(f"\n>> 저장: bi4_derived_cohort_intensity.csv")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: 파생 피처 포함 KMeans 재클러스터링
# ─────────────────────────────────────────────────────────────────────────────
header("STEP 6: 파생 피처 포함 확장 KMeans 클러스터링")

# 샘플링 (30K)
np.random.seed(42)
df_s = df.sample(n=min(30000, len(df)), random_state=42).copy().reset_index(drop=True)

# Encoding: 범주형 파생 컬럼 포함
le = LabelEncoder()
cat_derived = ['서비스구분', '성별', '지역권역', '상담강도_구간', '초기CO위험도']
if '출생코호트' in df_s.columns:
    cat_derived.append('출생코호트')

X_cat = pd.DataFrame()
for c in cat_derived:
    X_cat[c + '_enc'] = le.fit_transform(df_s[c].astype(str).str.strip().fillna('미상'))
print(f"\nCAT 파생 피처 서브스페이스: {list(X_cat.columns)}")

# 수치형 파생 컬럼 포함
num_derived = ['상담횟수_num', '총성공단계',
               '4주평가성공여부_bin', '6주평가성공여부_bin',
               '12주평가성공여부_bin', '6개월평가성공여부_bin']
X_num_raw = pd.DataFrame()
for c in num_derived:
    X_num_raw[c] = pd.to_numeric(df_s[c], errors='coerce').fillna(0)

co_col_num = co_col + '_num'
X_num_raw['초기CO값'] = df_s[co_col_num].fillna(0)

# df_s에도 동기화 (groupby 시 필요)
df_s['초기CO값'] = X_num_raw['초기CO값'].values

scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num_raw)
print(f"NUM 파생 피처 서브스페이스: {list(X_num_raw.columns)}")

# ── CAT 서브스페이스 클러스터링
print("\n[CAT 서브스페이스 — 파생 피처 포함]")
sil_cat = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_cat)
    sil_cat.append(silhouette_score(X_cat, lbl, sample_size=5000))
best_k_cat = K_range[np.argmax(sil_cat)]
print(f"CAT 실루엣 점수: {[round(s,3) for s in sil_cat]}")
print(f"최적 k (CAT) = {best_k_cat}")

km_cat = KMeans(n_clusters=best_k_cat, random_state=42, n_init=10)
df_s['cluster_cat_derived'] = km_cat.fit_predict(X_cat)

cat_profile = df_s.groupby('cluster_cat_derived')[cat_derived + ['상담강도_구간', '금연달성등급']].agg(
    lambda x: x.value_counts().index[0] if len(x) > 0 else 'N/A'
).reset_index()
counts = df_s['cluster_cat_derived'].value_counts().reset_index()
counts.columns = ['cluster_cat_derived', 'n']
cat_profile = cat_profile.merge(counts, on='cluster_cat_derived')

# 성공률 계산
eval_bin_cols = [c + '_bin' for c in eval_cols]
success_by_cat = df_s.groupby('cluster_cat_derived')[eval_bin_cols + ['총성공단계']].mean().round(3)
success_by_cat.columns = eval_cols + ['평균성공단계']
cat_profile = cat_profile.merge(success_by_cat.reset_index(), on='cluster_cat_derived')
print("\n[CAT 파생 피처 클러스터 프로파일]")
print(cat_profile.to_string(index=False))
cat_profile.to_csv(os.path.join(OUT, 'bi4_derived_cat_cluster.csv'), index=False, encoding='utf-8-sig')

# ── NUM 서브스페이스 클러스터링
print("\n[NUM 서브스페이스 — 파생 피처 포함]")
sil_num = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_num_scaled)
    sil_num.append(silhouette_score(X_num_scaled, lbl, sample_size=5000))
best_k_num = K_range[np.argmax(sil_num)]
print(f"NUM 실루엣 점수: {[round(s,3) for s in sil_num]}")
print(f"최적 k (NUM) = {best_k_num}")

km_num = KMeans(n_clusters=best_k_num, random_state=42, n_init=10)
df_s['cluster_num_derived'] = km_num.fit_predict(X_num_scaled)

num_profile = df_s.groupby('cluster_num_derived')[list(X_num_raw.columns)].mean().round(3)
counts_n = df_s['cluster_num_derived'].value_counts().reset_index()
counts_n.columns = ['cluster_num_derived', 'n']
num_profile = num_profile.reset_index().merge(counts_n, on='cluster_num_derived').sort_values('총성공단계', ascending=False)

# 레이블링
def label_cluster(row):
    s = row['총성공단계']
    c = row['상담횟수_num']
    co = row['초기CO값']
    if s >= 3.5:
        return '완전금연·고강도형'
    elif s >= 2.5 and c >= 10:
        return '부분성공·고강도형'
    elif s >= 2.5:
        return '부분성공·저강도형'
    elif s >= 1.0 and co >= 10:
        return '초기성공·고CO위험형'
    elif s < 1.0 and c >= 10:
        return '미달성·고강도집중형'
    else:
        return '미달성·저강도형'

num_profile['클러스터의미'] = num_profile.apply(label_cluster, axis=1)
print("\n[NUM 파생 피처 클러스터 프로파일]")
print(num_profile.to_string(index=False))
num_profile.to_csv(os.path.join(OUT, 'bi4_derived_num_cluster.csv'), index=False, encoding='utf-8-sig')

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: 요약 통계 — 파생 컬럼 활용 인사이트 종합
# ─────────────────────────────────────────────────────────────────────────────
header("STEP 7: 파생 컬럼 인사이트 종합 요약")

print("\n▶ [상담강도별 완전달성 비율 TOP]")
svc_grade = df.groupby('상담강도_구간').agg(
    총인원=('총성공단계', 'count'),
    완전달성=('총성공단계', lambda x: (x==4).sum()),
    미달성=('총성공단계', lambda x: (x==0).sum()),
    평균성공단계=('총성공단계', 'mean')
).reset_index()
svc_grade['완전달성률(%)'] = (svc_grade['완전달성'] / svc_grade['총인원'] * 100).round(2)
svc_grade['미달성률(%)'] = (svc_grade['미달성'] / svc_grade['총인원'] * 100).round(2)
print(svc_grade.to_string(index=False))

print("\n▶ [지역권역별 완전달성 비율]")
region_grade = df.groupby('지역권역').agg(
    총인원=('총성공단계', 'count'),
    완전달성=('총성공단계', lambda x: (x==4).sum()),
    평균성공단계=('총성공단계', 'mean')
).reset_index()
region_grade['완전달성률(%)'] = (region_grade['완전달성'] / region_grade['총인원'] * 100).round(2)
print(region_grade.to_string(index=False))

print("\n▶ [초기CO위험도별 완전달성 비율]")
co_grade = df.groupby('초기CO위험도').agg(
    총인원=('총성공단계', 'count'),
    완전달성=('총성공단계', lambda x: (x==4).sum()),
    평균성공단계=('총성공단계', 'mean')
).reset_index()
co_grade['완전달성률(%)'] = (co_grade['완전달성'] / co_grade['총인원'] * 100).round(2)
print(co_grade.to_string(index=False))

# 요약 저장
summary = {
    '상담강도별_완전달성': svc_grade.to_dict('records'),
    '지역권역별_완전달성': region_grade.to_dict('records'),
    'CO위험도별_완전달성': co_grade.to_dict('records'),
    'CAT_최적k': int(best_k_cat),
    'CAT_실루엣': round(float(max(sil_cat)), 3),
    'NUM_최적k': int(best_k_num),
    'NUM_실루엣': round(float(max(sil_num)), 3),
}
import json
with open(os.path.join(OUT, 'bi4_derived_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

svc_grade.to_csv(os.path.join(OUT, 'bi4_derived_intensity_summary.csv'), index=False, encoding='utf-8-sig')
region_grade.to_csv(os.path.join(OUT, 'bi4_derived_region_summary.csv'), index=False, encoding='utf-8-sig')
co_grade.to_csv(os.path.join(OUT, 'bi4_derived_co_summary.csv'), index=False, encoding='utf-8-sig')

print(f"\n\n{'='*72}")
print(f"  BI-4 파생피처 분析 완료 — 출력 파일 목록:")
print(f"{'='*72}")
for f in sorted(os.listdir(OUT)):
    if 'bi4' in f:
        print(f"  {f}")
