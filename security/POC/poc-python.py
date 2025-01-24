from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer, util
import faiss
import json

# 1. 보안 가이드 데이터를 로드
with open("security_guidelines.json", "r", encoding="utf-8") as file:
    guidelines = json.load(file)

# 2. 문서 임베딩 준비 (SentenceTransformer)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
guideline_texts = [guide["guideline"] for guide in guidelines]
guideline_embeddings = embedding_model.encode(guideline_texts, convert_to_tensor=True)

# 3. FAISS를 사용하여 가이드라인 인덱스 구축
index = faiss.IndexFlatL2(guideline_embeddings.shape[1])
index.add(guideline_embeddings.cpu().numpy())

# 4. 소스 코드 샘플
source_code = """
user_input = input("User: ")  # 사용자 입력
prompt = user_input  # 입력을 그대로 모델에 전달
response = model.generate(prompt)
print(response)
"""

# 5. 소스 코드 분석
def analyze_code(code_snippet):
    # 소스 코드에서 텍스트 임베딩 생성
    code_embedding = embedding_model.encode(code_snippet, convert_to_tensor=True)
    
    # 가장 가까운 가이드라인 검색
    distances, indices = index.search(code_embedding.cpu().numpy(), k=1)
    closest_index = indices[0][0]
    closest_guide = guidelines[closest_index]
    
    # 출력 결과
    return {
        "code_snippet": code_snippet.strip(),
        "guideline": closest_guide["guideline"],
        "section": closest_guide["section"],
        "distance": distances[0][0]
    }

# 6. 실행 테스트
result = analyze_code(source_code)
print("Code Snippet:", result["code_snippet"])
print("Closest Guideline:", result["guideline"])
print("Section:", result["section"])
print("Distance:", result["distance"])
