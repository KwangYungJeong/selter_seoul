import pandas as pd
import numpy as np
import sys

# 윈도우 터미널 UTF-8 출력 설정
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

def run_eda(csv_path):
    print(f"--- 데이터 탐색적 분석(EDA) ---")
    print(f"대상 파일: {csv_path}\n")
    
    # 변환된 CSV 파일 읽기
    df = pd.read_csv(csv_path)
    
    total_original = len(df)
    original_columns = list(df.columns)
    
    # 결측치 및 조건에 맞는 데이터 필터링 (shelter_ml.py와 동일한 조건)
    active_df = df[df['해제일자'].isna()].copy()
    active_df = active_df[active_df['최대수용인원'].notna() & (active_df['최대수용인원'] > 0)]
    
    # 남길 컬럼 추출
    coords_df = active_df[['위도(EPSG4326)', '경도(EPSG4326)', '최대수용인원']].dropna()
    
    total_retained = len(coords_df)
    total_dropped = total_original - total_retained
    retained_columns = list(coords_df.columns)
    
    # 1. 데이터 Drop 정보
    print("1. 데이터 필터링 현황")
    print(f" - 전체 데이터 개수: {total_original}개")
    print(f" - 사용 가능한 데이터(유지): {total_retained}개")
    print(f" - 제외된 데이터(Drop): {total_dropped}개")
    print("   (해제일자가 존재하거나 최대수용인원이 없는/0인 데이터 제외됨)\n")
    
    # 2. 컬럼 정보
    print("2. 컬럼 정보")
    print(f" - 원본 컬럼 리스트 (총 {len(original_columns)}개):")
    print(f"   {original_columns}")
    print(f" - 학습에 남긴 컬럼 리스트 (총 {len(retained_columns)}개):")
    print(f"   {retained_columns}\n")
    
    # 3. 수용인원 통계치
    print("3. 수용인원 통계치 (EDA)")
    capacity_stats = coords_df['최대수용인원'].describe()
    print(f" - 개수: {capacity_stats['count']:.0f}개")
    print(f" - 평균: {capacity_stats['mean']:.1f}명")
    print(f" - 표준편차: {capacity_stats['std']:.1f}")
    print(f" - 최소값(Min): {capacity_stats['min']:.0f}명")
    print(f" - 1사분위(25%): {capacity_stats['25%']:.0f}명")
    print(f" - 중앙값(50%): {capacity_stats['50%']:.0f}명")
    print(f" - 3사분위(75%): {capacity_stats['75%']:.0f}명")
    print(f" - 최대값(Max): {capacity_stats['max']:.0f}명")

if __name__ == "__main__":
    # 상대경로 사용
    csv_file = 'shelter_seoul.csv'
    run_eda(csv_file)
