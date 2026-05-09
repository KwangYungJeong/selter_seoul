import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from shelter_ml import create_shelter_models
import warnings
warnings.filterwarnings('ignore') # 일부 matplotlib 경고 무시

def visualize_random_forest_trees():
    print("--- Decision Tree 시각화 스크립트 시작 ---")
    # 기존 모듈 재사용하여 모델 학습
    models = create_shelter_models('shelter_seoul.csv')
    rf_model = models['RandomForest']
    
    print("\n[시각화] Random Forest에서 첫 5개의 Decision Tree를 추출합니다...")
    
    # 1행 5열의 그래프 공간 준비 (가로로 길게)
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(28, 6))
    
    # Random Forest 내부의 여러 트리(estimators) 중 처음 5개 선택
    estimators = rf_model.estimators_[:5]
    
    for index, tree_model in enumerate(estimators):
        # 트리가 너무 깊으면 글씨가 뭉쳐서 안 보이므로, 상위 3단계(max_depth=3)까지만 렌더링합니다.
        plot_tree(tree_model, 
                  feature_names=['Latitude', 'Longitude'], 
                  class_names=['Danger(0)', 'Safe(1)'], 
                  filled=True, 
                  ax=axes[index],
                  max_depth=3, # 시각화 시 가독성을 위한 깊이 제한
                  fontsize=7,
                  rounded=True)
        
        axes[index].set_title(f'Tree #{index+1}', fontsize=14, fontweight='bold')
        
    plt.tight_layout()
    output_path = 'rf_5_trees.png'
    plt.savefig(output_path, dpi=300)
    print(f"\n성공! 5개의 트리가 시각화되어 '{output_path}' 이미지 파일로 저장되었습니다.")

if __name__ == "__main__":
    visualize_random_forest_trees()
