import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

def create_shelter_models(csv_path):
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
    
    # [수정] 수용인원 기반 동적 반경 계산 (기본 50m, 1000명당 15m 추가, 최대 500m 제한)
    radii_m = np.clip(50 + (capacities / 1000.0) * 15, 50, 500)
    
    # Positive Data (대피소 실제 위치, 라벨 1)
    positive_data = coords
    y_pos = np.ones(len(positive_data))
    print(f"   -> 활성 민방위 대피소 데이터 갯수 (Positive): {len(positive_data)}개")
    
    # Negative Data (가짜 위치, 라벨 0) 생성
    print("2. Negative Data (실질적 대피소 커버리지 밖의 구역) 생성 중...")
    start_time = time.time()
    
    tree = BallTree(np.radians(positive_data), metric='haversine')
    earth_radius_m = 6371000.0
    radii_rad = radii_m / earth_radius_m  # 각 대피소별 라디안 반경
    
    negative_data = []
    target_neg_count = len(positive_data)
    
    # k=10을 사용하여 주변 10개의 대피소를 모두 검사하여 겹침 방지
    while len(negative_data) < target_neg_count:
        random_lats = np.random.uniform(lat_min, lat_max, 100000)
        random_lngs = np.random.uniform(lng_min, lng_max, 100000)
        random_coords = np.column_stack((random_lats, random_lngs))
        
        # 각 랜덤 좌표에 대해 가장 가까운 10개의 대피소 거리와 인덱스 추출
        distances, indices = tree.query(np.radians(random_coords), k=10)
        
        # 특정 랜덤 좌표가 주변 10개 대피소 중 "단 하나라도" 해당 대피소의 동적 반경 안에 들어가면 커버되는 것으로 간주
        is_covered = distances <= radii_rad[indices]
        point_covered = np.any(is_covered, axis=1)
        
        valid_negatives = random_coords[~point_covered]
        negative_data.extend(valid_negatives.tolist())
    
    negative_data = np.array(negative_data[:target_neg_count])
    y_neg = np.zeros(len(negative_data))
    print(f"   -> 무작위 취약 구역 위치 갯수 (Negative): {len(negative_data)}개 (소요 시간: {time.time()-start_time:.2f}초)")
    
    print("\n3. 머신러닝 모델 학습 및 비교")
    X = np.vstack((positive_data, negative_data))
    y = np.concatenate((y_pos, y_neg))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {}
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
    models['RandomForest'] = rf_model
    print(f" [Random Forest] 정확도: {rf_acc:.4f}")
    
    # KNN
    k_values = [1, 5, 15]
    for k in k_values:
        knn_model = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1)
        knn_model.fit(X_train, y_train)
        knn_acc = accuracy_score(y_test, knn_model.predict(X_test))
        models[f'KNN(K={k})'] = knn_model
        print(f" [KNN (K={k:2d})]    정확도: {knn_acc:.4f}")
        
    return models

def predict_and_compare(models, lat, lng, location_name):
    print(f"\n=== 테스트 위치: {location_name} ===")
    print(f"(위도: {lat:.5f}, 경도: {lng:.5f}) 주변에 적절한 대피소가 있을(안전도) 확률:")
    
    for name, model in models.items():
        prob = model.predict_proba([[lat, lng]])
        shelter_prob = prob[0][1] * 100
        print(f" - {name:<15} : {shelter_prob:5.1f}%")

if __name__ == "__main__":
    file_path = 'shelter_seoul.csv'
    try:
        models = create_shelter_models(file_path)
        print("\n=======================================================")
        
        # 테스트 1: 대형 대피소가 모여있을 법한 서울시청 부근
        test_lat1, test_lng1 = 37.5665, 126.9780 
        predict_and_compare(models, test_lat1, test_lng1, "서울시청 부근 (도심 지역)")
        
        # 테스트 2: 산지나 한강 등 대피소가 없을 만한 무작위 위치
        test_lat2, test_lng2 = 37.5500, 126.9000 
        predict_and_compare(models, test_lat2, test_lng2, "외곽 무작위 위치")
        
    except Exception as e:
        print(f"오류 발생: {e}")


