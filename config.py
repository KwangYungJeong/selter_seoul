# 대피소 수용 인원에 따른 동적 커버리지 반경 계산 하이퍼파라미터

# 기본 커버리지 반경 (m)
MIN_RADIUS_M_LIST = [30.0, 50.0]

# 수용 인원 기준 단위 (명)
CAPACITY_BASE_UNIT_LIST = [500.0, 1000.0]

# 기준 단위당 추가되는 커버리지 반경 (m)
RADIUS_PER_BASE_UNIT_M_LIST = [10.0, 15.0, 20.0]

# 최대 커버리지 반경 제한 (m)
MAX_RADIUS_M_LIST = [300.0, 500.0, 1000.0]

# --- 2. 지도 시각화용 수동(고정) 파라미터 (최적화 결과 반영) ---
MANUAL_MIN_RADIUS_M = 50.0
MANUAL_CAPACITY_BASE_UNIT = 500.0
MANUAL_RADIUS_PER_BASE_UNIT_M = 20.0
MANUAL_MAX_RADIUS_M = 1000.0

import numpy as np

def calculate_dynamic_radius(capacities, min_r=MANUAL_MIN_RADIUS_M, base_u=MANUAL_CAPACITY_BASE_UNIT, per_base=MANUAL_RADIUS_PER_BASE_UNIT_M, max_r=MANUAL_MAX_RADIUS_M):
    """
    수용 인원 배열을 기반으로 동적 커버리지 반경(m)을 계산합니다.
    """
    return np.clip(min_r + (capacities / base_u) * per_base, min_r, max_r)


# --- 머신러닝 모델 하이퍼파라미터 ---

# Random Forest
RF_N_ESTIMATORS_LIST = [20, 50, 100, 200]
RF_MAX_DEPTH_LIST = [3, 5, 7, 9]

# KNN (테스트할 K 값들의 리스트: 1~50 사이의 소수, 1포함 2제외)
KNN_K_VALUES = [1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
