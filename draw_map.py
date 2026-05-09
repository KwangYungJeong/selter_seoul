import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shelter_ml import create_shelter_models

def draw_map(csv_path, output_path):
    print("\n--- 지도 생성 스크립트 시작 ---")
    print("1. 데이터를 불러오고 모델을 재학습합니다...")
    # 기존 코드의 학습 함수 재활용
    models = create_shelter_models(csv_path)
    rf_model = models['RandomForest']
    
    print("\n2. 대피소 위치 데이터를 불러옵니다...")
    df = pd.read_csv(csv_path)
    active_df = df[df['해제일자'].isna()].copy()
    active_df = active_df[active_df['최대수용인원'].notna() & (active_df['최대수용인원'] > 0)]
    shelter_lats = active_df['위도(EPSG4326)'].values
    shelter_lngs = active_df['경도(EPSG4326)'].values
    
    print("3. 서울 전역의 가상 좌표 격자(Grid)를 생성합니다...")
    lat_min, lat_max = 37.42, 37.70
    lng_min, lng_max = 126.80, 127.18
    
    # 가로세로 100칸 (총 10,000개의 위치)
    lats = np.linspace(lat_min, lat_max, 100)
    lngs = np.linspace(lng_min, lng_max, 100)
    grid_lng, grid_lat = np.meshgrid(lngs, lats)
    
    grid_points = np.c_[grid_lat.ravel(), grid_lng.ravel()]
    
    print("4. 생성된 10,000개의 위치 각각에 대해 대피소 안전도 예측을 수행합니다...")
    # 확률 예측 (1에 가까울수록 안전, 0에 가까울수록 취약)
    probs = rf_model.predict_proba(grid_points)[:, 1]
    
    print("5. 그림(지도)을 생성하여 저장합니다...")
    plt.figure(figsize=(14, 10))
    
    # 1) 격자 좌표 플롯 (색상 맵: RdYlGn - Red: 취약, Green: 안전)
    sc = plt.scatter(grid_points[:, 1], grid_points[:, 0], c=probs, cmap='RdYlGn', s=50, alpha=0.6, marker='s')
    
    # 2) 실제 활성 대피소 위치 오버레이 (작은 파란 점)
    plt.scatter(shelter_lngs, shelter_lats, c='blue', s=3, alpha=0.5, label='Actual Shelters (Blue dots)')
    
    # 색상바 추가
    cbar = plt.colorbar(sc)
    cbar.set_label('Safety Probability (0.0 = Vulnerable/Blind Spot, 1.0 = Highly Safe)', fontsize=12)
    
    plt.title('Seoul Civil Defense Shelter Coverage Map\n(Red areas represent Blind Spots lacking adequate shelter capacity)', fontsize=16, pad=20)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.xlim(lng_min, lng_max)
    plt.ylim(lat_min, lat_max)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n성공적으로 지도를 그려 저장했습니다: {output_path}")

if __name__ == "__main__":
    csv_path = 'shelter_seoul.csv'
    output_path = '/Users/kyle/.gemini/antigravity/brain/a58f20b6-766c-4594-891a-8288cff529d4/blind_spots.png'
    draw_map(csv_path, output_path)
