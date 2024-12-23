import numpy as np
import faiss

# 랜덤 데이터 생성 (1000개의 128차원 벡터)
dimension = 128
num_vectors = 1000
query_vectors = 5

# 데이터베이스 벡터 생성
np.random.seed(1234)
database_vectors = np.random.random((num_vectors, dimension)).astype('float32')

# 검색할 쿼리 벡터 생성
query = np.random.random((query_vectors, dimension)).astype('float32')

# FAISS 인덱스 생성 (L2 거리 기반)
index = faiss.IndexFlatL2(dimension)

# 데이터 추가
index.add(database_vectors)

# 쿼리 벡터 검색
k = 5  # 반환할 최근접 이웃의 수
distances, indices = index.search(query, k)

# 결과 출력
print("Query Results:")
for i in range(query_vectors):
    print(f"Query Vector {i}:")
    print(f"Indices: {indices[i]}")
    print(f"Distances: {distances[i]}")
