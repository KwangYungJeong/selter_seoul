import numpy as np
import pandas as pd
import folium
import matplotlib.colors as colors
import os
from config import calculate_dynamic_radius
from shelter_ml import create_shelter_models

def draw_satellite_map(csv_path, output_path):
    print("\n--- 위성 지도 기반 시각화 시작 ---")
    print("1. 대피소 위치 데이터를 불러옵니다...")
    df = pd.read_csv(csv_path)
    active_df = df[df['해제일자'].isna()].copy()
    active_df = active_df[active_df['최대수용인원'].notna() & (active_df['최대수용인원'] > 0)]
    active_df = active_df.dropna(subset=['위도(EPSG4326)', '경도(EPSG4326)'])
    shelter_lats = active_df['위도(EPSG4326)'].values
    shelter_lngs = active_df['경도(EPSG4326)'].values
    shelter_caps = active_df['최대수용인원'].values
    
    print("2. 인터랙티브 위성 지도를 렌더링합니다...")
    # 서울시청 중심의 지도 생성
    m = folium.Map(location=[37.5665, 126.9780], zoom_start=11, control_scale=True)
    
    # Esri 위성 지도 레이어 추가 (구글 어스와 유사한 고해상도 위성지도)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # 기본 지도 레이어도 유지
    folium.TileLayer('openstreetmap', name='OpenStreetMap', overlay=False).add_to(m)
    
    print(" -> 수용 인원에 따른 대피소 커버리지(반투명 안전구역) 그리는 중...")
    # 실제 대피소 위치 추가 (수용 인원에 비례한 실제 커버리지 반경 포함)
    for lat, lng, cap in zip(shelter_lats, shelter_lngs, shelter_caps):
        # 1. 실제 커버리지 반경 계산 (최적화된 파라미터 적용)
        radius_m = float(calculate_dynamic_radius(np.array([cap]))[0])
        
        # 2. 커버리지 반경을 투명한 밝은 녹색(안전) 원으로 추가
        folium.Circle(
            location=[lat, lng],
            radius=radius_m,
            color='#00FF00', # 눈에 띄는 밝은 녹색 테두리
            weight=1.5,
            fill=True,
            fill_color='#00FF00', # 내부 녹색 채우기
            fill_opacity=0.3 # 위성 지도가 보이는 반투명
        ).add_to(m)
        
        # 3. 중심점(대피소 위치)
        folium.CircleMarker(
            location=[lat, lng],
            radius=1.5,
            color='#00BFFF', # 밝은 파란색
            fill=True,
            fill_opacity=0.9,
            tooltip=f"수용 인원: {int(cap):,}명<br>커버리지 반경: {int(radius_m)}m"
        ).add_to(m)
        
    # 레이어 컨트롤 추가
    folium.LayerControl().add_to(m)
    
    m.save(output_path)
    print(f"\n완료! HTML 지도가 생성되었습니다: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    csv_path = 'shelter_seoul.csv'
    output_path = 'shelter_satellite_map.html'
    draw_satellite_map(csv_path, output_path)
