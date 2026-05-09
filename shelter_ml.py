import sys
import os

# 윈도우 터미널 UTF-8 출력 설정
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python 3.7 미만 버전 대응
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import itertools
from config import (
    calculate_dynamic_radius, 
    MIN_RADIUS_M_LIST, CAPACITY_BASE_UNIT_LIST, RADIUS_PER_BASE_UNIT_M_LIST, MAX_RADIUS_M_LIST,
    MANUAL_MIN_RADIUS_M, MANUAL_CAPACITY_BASE_UNIT, MANUAL_RADIUS_PER_BASE_UNIT_M, MANUAL_MAX_RADIUS_M,
    RF_N_ESTIMATORS_LIST, 
    RF_MAX_DEPTH_LIST, 
    KNN_K_VALUES
)

def create_shelter_models(csv_path, manual_only=False):
    print("1. 데이터 로드 및 전처리 중...")
    df = pd.read_csv(csv_path)
    
    # [수정] 해제일자가 있는(이미 취소된) 대피소 제외 및 수용인원 이상치 제거
    active_df = df[df['해제일자'].isna()].copy()
    active_df = active_df[active_df['최대수용인원'].notna() & (active_df['최대수용인원'] > 0)]
    
    # 위도, 경도, 수용인원 컬럼 추출
    coords_df = active_df[['위도(EPSG4326)', '경도(EPSG4326)', '최대수용인원']].dropna()
    coords_df.columns = ['lat', 'lng', 'capacity']
    
    # 서울시 대략적 위경도 범위 내의 데이터만 필터링 (이상치 제거)
    lat_min, lat_max = coords_df['lat'].min(), coords_df['lat'].max()
    lng_min, lng_max = coords_df['lng'].min(), coords_df['lng'].max()
    
    lat_min, lat_max = max(37.4, lat_min), min(37.7, lat_max)
    lng_min, lng_max = max(126.8, lng_min), min(127.2, lng_max)
    
    valid_mask = (coords_df['lat'] >= lat_min) & (coords_df['lat'] <= lat_max) & \
                 (coords_df['lng'] >= lng_min) & (coords_df['lng'] <= lng_max)
    coords_df = coords_df[valid_mask]
    
    coords = coords_df[['lat', 'lng']].values
    capacities = coords_df['capacity'].values
    
    positive_data = coords
    y_pos = np.ones(len(positive_data))
    print(f"   -> 활성 민방위 대피소 데이터 갯수 (Positive): {len(positive_data)}개")
    
    tree = BallTree(np.radians(positive_data), metric='haversine')
    earth_radius_m = 6371000.0
    
    if manual_only:
        radius_combinations = [(MANUAL_MIN_RADIUS_M, MANUAL_CAPACITY_BASE_UNIT, MANUAL_RADIUS_PER_BASE_UNIT_M, MANUAL_MAX_RADIUS_M)]
    else:
        radius_combinations = list(itertools.product(
            MIN_RADIUS_M_LIST, CAPACITY_BASE_UNIT_LIST, RADIUS_PER_BASE_UNIT_M_LIST, MAX_RADIUS_M_LIST
        ))

    
    print(f"\n2. Dataset and Parameter Search ({len(radius_combinations)} combinations)")
    
    global_best_rf_acc = 0
    global_best_rf_config = None
    global_best_knn_acc = 0
    global_best_knn_config = None
    
    global_best_models = {}
    all_results = []
    rf_results_detailed = []
    knn_results_detailed = []
    
    for r_idx, (min_r, base_u, per_base, max_r) in enumerate(radius_combinations):
        print(f"\n--- [조합 {r_idx+1}/{len(radius_combinations)}] min={min_r}, base={base_u}, per={per_base}, max={max_r} ---")
        start_time = time.time()
        
        # 반경 계산
        radii_m = calculate_dynamic_radius(capacities, min_r, base_u, per_base, max_r)
        radii_rad = radii_m / earth_radius_m
        
        # Negative Data (실질적 대피소 커버리지 밖의 구역) 생성
        np.random.seed(42)  # 재현성을 위한 시드 고정
        negative_data = []
        target_neg_count = len(positive_data)
        
        while len(negative_data) < target_neg_count:
            random_lats = np.random.uniform(lat_min, lat_max, 100000)
            random_lngs = np.random.uniform(lng_min, lng_max, 100000)
            random_coords = np.column_stack((random_lats, random_lngs))
            
            distances, indices = tree.query(np.radians(random_coords), k=10)
            is_covered = distances <= radii_rad[indices]
            point_covered = np.any(is_covered, axis=1)
            
            valid_negatives = random_coords[~point_covered]
            negative_data.extend(valid_negatives.tolist())
        
        negative_data = np.array(negative_data[:target_neg_count])
        y_neg = np.zeros(len(negative_data))
        
        X = np.vstack((positive_data, negative_data))
        y = np.concatenate((y_pos, y_neg))
        
        # [추가] 최종 데이터셋 구성 정보 출력
        if r_idx == 0:
            print(f"\n--- 최종 데이터셋 구성 및 파생 컬럼 확인 ---")
            print(f"   -> Positive(대피소): {len(positive_data)}개")
            print(f"   -> Negative(사각지대): {len(negative_data)}개")
            print(f"   -> 총 데이터 개수: {len(X)}개")
            
            # 컬럼 구조 시각화용 출력 (설명과 일치시키기 위해 파생 컬럼 포함)
            # Positive 데이터에 대해 샘플 생성 (Radius가 존재하므로)
            preview_df = pd.DataFrame(positive_data[:5], columns=['Latitude', 'Longitude'])
            preview_df['Capacity'] = capacities[:5]
            preview_df['Radius(m)'] = radii_m[:5]
            preview_df['Label'] = 1.0 # Positive sample
            
            print(f"   -> 사용 및 생성된 주요 컬럼: {list(preview_df.columns)}")
            print(f"   -> 데이터셋 샘플 (상위 5개, 대피소 기준):\n{preview_df.to_string(index=False)}\n")
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        
        # --- ML 탐색 ---
        best_rf = None
        best_rf_acc = 0
        best_rf_name = ""
        for n_est in RF_N_ESTIMATORS_LIST:
            for depth in RF_MAX_DEPTH_LIST:
                rf_model = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1)
                rf_model.fit(X_train, y_train)
                acc = accuracy_score(y_test, rf_model.predict(X_test))
                
                # 시각화용 상세 데이터 수집 (RF)
                rf_results_detailed.append({
                    'min_r': min_r, 'base_u': base_u, 'per_base': per_base, 'max_r': max_r,
                    'n_estimators': n_est, 'max_depth': depth, 'accuracy': acc
                })
                
                if acc > best_rf_acc:
                    best_rf_acc = acc
                    best_rf = rf_model
                    best_rf_name = f'RF(n={n_est}, d={depth})'
        
        best_knn = None
        best_knn_acc = 0
        best_knn_name = ""
        for k in KNN_K_VALUES:
            knn_model = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1)
            knn_model.fit(X_train, y_train)
            acc = accuracy_score(y_test, knn_model.predict(X_test))
            
            # 시각화용 상세 데이터 수집 (KNN)
            knn_results_detailed.append({
                'min_r': min_r, 'base_u': base_u, 'per_base': per_base, 'max_r': max_r,
                'K': k, 'accuracy': acc
            })
            
            if acc > best_knn_acc:
                best_knn_acc = acc
                best_knn = knn_model
                best_knn_name = f'KNN(K={k})'
                
        print(f" -> Time: {time.time()-start_time:.1f}s | {best_rf_name}: {best_rf_acc:.4f} | {best_knn_name}: {best_knn_acc:.4f}")
        
        # 결과 저장을 위한 데이터 수집 (RF/KNN 중 더 좋은 쪽의 정확도 기록)
        result_entry = {
            'min_r': min_r, 'base_u': base_u, 'per_base': per_base, 'max_r': max_r,
            'best_rf_params': best_rf_name, 'best_rf_acc': best_rf_acc,
            'best_knn_params': best_knn_name, 'best_knn_acc': best_knn_acc,
            'winner_acc': max(best_rf_acc, best_knn_acc)
        }
        all_results.append(result_entry)

        # 개별 모델별로 최고 성능일 때의 파라미터 조합 저장
        if best_rf_acc > global_best_rf_acc:
            global_best_rf_acc = best_rf_acc
            global_best_rf_config = (min_r, base_u, per_base, max_r)
            global_best_models['Best_RF'] = best_rf
            global_best_models['Best_RF_Name'] = best_rf_name
            
            # [추가] 상세 지표 계산
            y_pred = best_rf.predict(X_test)
            global_best_models['RF_Metrics'] = {
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'cm': confusion_matrix(y_test, y_pred)
            }

        if best_knn_acc > global_best_knn_acc:
            global_best_knn_acc = best_knn_acc
            global_best_knn_config = (min_r, base_u, per_base, max_r)
            global_best_models['Best_KNN'] = best_knn
            global_best_models['Best_KNN_Name'] = best_knn_name

            # [추가] 상세 지표 계산
            y_pred = best_knn.predict(X_test)
            global_best_models['KNN_Metrics'] = {
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'cm': confusion_matrix(y_test, y_pred)
            }

    # 모든 결과를 CSV로 저장
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('ml_results.csv', index=False, encoding='utf-8-sig')
    
    # 상세 결과 저장 (시각화용)
    pd.DataFrame(rf_results_detailed).to_csv('rf_detailed_results.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(knn_results_detailed).to_csv('knn_detailed_results.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n[Success] All results saved (ml_results.csv, rf_detailed_results.csv, knn_detailed_results.csv)")

    print(f"\n=======================================================")
    print(f"[Search Complete - Detailed Performance Report]")
    
    rf_m = global_best_models.get('RF_Metrics')
    if rf_m:
        print(f"\n[Best RF: {global_best_models['Best_RF_Name']}]")
        print(f" - Accuracy:  {global_best_rf_acc:.4f}")
        print(f" - Precision: {rf_m['precision']:.4f}")
        print(f" - Recall:    {rf_m['recall']:.4f}")
        print(f" - F1-Score:  {rf_m['f1']:.4f}")
        print(f" - Confusion Matrix:\n{rf_m['cm']}")

    knn_m = global_best_models.get('KNN_Metrics')
    if knn_m:
        print(f"\n[Best KNN: {global_best_models['Best_KNN_Name']}]")
        print(f" - Accuracy:  {global_best_knn_acc:.4f}")
        print(f" - Precision: {knn_m['precision']:.4f}")
        print(f" - Recall:    {knn_m['recall']:.4f}")
        print(f" - F1-Score:  {knn_m['f1']:.4f}")
        print(f" - Confusion Matrix:\n{knn_m['cm']}")
    
    print(f"=======================================================")
    
    return global_best_models, global_best_rf_config


def predict_and_compare(models, lat, lng, location_name):
    print(f"\n=== Test Location: {location_name} ===")
    print(f"(Lat: {lat:.5f}, Lng: {lng:.5f}) Safety Probability:")
    
    X_test = np.array([[lat, lng]])
    
    rf_name = models.get('Best_RF_Name', 'RF')
    rf_model = models.get('Best_RF')
    if rf_model:
        prob = rf_model.predict_proba(X_test)[0][1]
        print(f" - {rf_name:<20}: {prob*100:>5.1f}%")

    knn_name = models.get('Best_KNN_Name', 'KNN')
    knn_model = models.get('Best_KNN')
    if knn_model:
        prob = knn_model.predict_proba(X_test)[0][1]
        print(f" - {knn_name:<20}: {prob*100:>5.1f}%")

if __name__ == "__main__":
    file_path = 'shelter_seoul.csv'
    try:
        models, best_config = create_shelter_models(file_path)
        
        # 테스트 1: 대형 대피소가 모여있을 법한 서울시청 부근
        test_lat1, test_lng1 = 37.5665, 126.9780 
        predict_and_compare(models, test_lat1, test_lng1, "서울시청 부근 (도심 지역)")
        
        # 테스트 2: 산지나 한강 등 대피소가 없을 만한 무작위 위치
        test_lat2, test_lng2 = 37.5500, 126.9000 
        predict_and_compare(models, test_lat2, test_lng2, "외곽 무작위 위치")
        
    except Exception as e:
        print(f"오류 발생: {e}")


