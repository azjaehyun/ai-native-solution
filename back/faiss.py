import numpy as np
import brew install cmake libomp

def test_faiss_and_numpy():
    try:
        # 샘플 데이터 생성 (3차원 벡터)
        sample_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
        
        # FAISS Index 생성 (L2 거리 기반)
        dimension = sample_data.shape[1]  # 벡터의 차원
        index = faiss.IndexFlatL2(dimension)
        
        # 데이터 추가
        index.add(sample_data)
        print("FAISS Index에 데이터 추가 완료.")
        
        # 검색 (Query 벡터)
        query_vector = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        distances, indices = index.search(query_vector, k=2)  # 가장 가까운 2개의 벡터 검색
        
        print("Query Vector:", query_vector)
        print("Closest Distances:", distances)
        print("Closest Indices:", indices)
        
        print("FAISS와 numpy가 정상적으로 동작합니다.")
    
    except Exception as e:
        print("오류 발생:", e)

# 함수 실행
if __name__ == "__main__":
    test_faiss_and_numpy()
