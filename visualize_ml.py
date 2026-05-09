import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# 윈도우 터미널 UTF-8 출력 설정
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def visualize():
    print("시각화 데이터를 불러오는 중...")
    
    if not os.path.exists('rf_detailed_results.csv') or not os.path.exists('knn_detailed_results.csv'):
        print("에러: ML 결과 파일(csv)이 존재하지 않습니다. shelter_ml.py를 먼저 실행해주세요.")
        return

    rf_df = pd.read_csv('rf_detailed_results.csv')
    knn_df = pd.read_csv('knn_detailed_results.csv')
    ml_res_df = pd.read_csv('ml_results.csv')

    # 1. Random Forest 하이퍼파라미터 히트맵
    # 모든 반경 조합에 대한 평균 정확도 계산
    rf_pivot = rf_df.pivot_table(index='max_depth', columns='n_estimators', values='accuracy', aggfunc='mean')
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(rf_pivot, annot=True, fmt=".4f", cmap='YlGnBu')
    plt.title('Random Forest Accuracy: n_estimators vs max_depth\n(Averaged over all radius combinations)')
    plt.savefig('rf_hyperparameter_heatmap.png', dpi=200)
    print("저장 완료: rf_hyperparameter_heatmap.png")

    # 2. KNN 하이퍼파라미터 추이
    knn_avg = knn_df.groupby('K')['accuracy'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    plt.plot(knn_avg['K'], knn_avg['accuracy'], marker='o', linestyle='-', color='orange')
    plt.title('KNN Accuracy Trend by K value\n(Averaged over all radius combinations)')
    plt.xlabel('K value')
    plt.ylabel('Mean Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('knn_accuracy_trend.png', dpi=200)
    print("저장 완료: knn_accuracy_trend.png")

    # 3. 반경 조합별 성능 비교 (Top 10)
    top_radius = ml_res_df.sort_values(by='winner_acc', ascending=False).head(10)
    top_radius['config_str'] = top_radius.apply(lambda x: f"m:{int(x['min_r'])} b:{int(x['base_u'])} p:{int(x['per_base'])} M:{int(x['max_r'])}", axis=1)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='winner_acc', y='config_str', data=top_radius, palette='viridis')
    plt.xlim(top_radius['winner_acc'].min() - 0.01, top_radius['winner_acc'].max() + 0.005)
    plt.title('Top 10 Radius Configurations (Best Model Accuracy)')
    plt.xlabel('Accuracy')
    plt.ylabel('Radius Configuration')
    plt.tight_layout()
    plt.savefig('radius_config_comparison.png', dpi=200)
    print("저장 완료: radius_config_comparison.png")

    print("\n[시각화 완료] 모든 그래프가 현재 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    visualize()
