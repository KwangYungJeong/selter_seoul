import pandas as pd
import os

def convert_csv_encoding(input_file, output_file):
    print(f"'{input_file}' 파일을 읽는 중입니다 (인코딩: cp949)...")
    try:
        # cp949 형식으로 데이터 읽기
        df = pd.read_csv(input_file, encoding='cp949')
        
        print(f"'{output_file}' 파일로 저장 중입니다 (인코딩: UTF-8)...")
        # utf-8 형식으로 저장 (인덱스 제외)
        df.to_csv(output_file, encoding='utf-8', index=False)
        
        print("변환이 성공적으로 완료되었습니다!")
        print(f"새 파일 크기: {os.path.getsize(output_file) / 1024:.2f} KB")
        
    except FileNotFoundError:
        print(f"에러: '{input_file}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"변환 중 에러가 발생했습니다: {e}")

if __name__ == "__main__":
    original_csv = '민방위대피시설_서울특별시.csv'
    new_csv = 'shelter_seoul.csv'
    
    convert_csv_encoding(original_csv, new_csv)
