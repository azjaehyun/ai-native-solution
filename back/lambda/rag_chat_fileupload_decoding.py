import boto3
import json
import faiss
import numpy as np
import base64
from typing import List

# Bedrock 클라이언트 생성
bedrock_agent_runtime = boto3.client(
    service_name="bedrock-agent-runtime",
    region_name="ap-northeast-2"  # 실제 사용하는 리전으로 변경
)

def extract_text_from_file(file_content):
    """첨부 파일에서 텍스트 추출 (예시: 텍스트 파일)."""
    print("Decoding file content...")
    return file_content.decode('utf-8')  # 간단히 디코딩

def generate_embeddings(text: List[str], model_id: str) -> List[np.ndarray]:
    """Amazon Bedrock을 사용하여 텍스트 임베딩 생성."""
    print(f"Generating embeddings for {len(text)} chunks using model {model_id}...")
    bedrock_runtime = boto3.client('bedrock-runtime', region_name='ap-northeast-2')
    embeddings = []

    for i, chunk in enumerate(text):
        try:
            print(f"[{i + 1}/{len(text)}] Processing chunk: {chunk[:50]}...")
            
            # 빈 문자열 또는 비정상적 텍스트 필터링
            if not chunk.strip():
                print(f"Skipping empty or invalid chunk at index {i}.")
                continue
            
            # Bedrock API 호출에 적합한 형식으로 요청 생성
            body = json.dumps({
                "prompt": f'"{chunk}"',  # 큰따옴표로 감싸기
                "max_tokens_to_sample": 200
            })
            
            # Bedrock 모델 호출
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                contentType='application/json',
                accept='application/json',
                body=body
            )
            
            # 응답 처리
            response_body = response['body'].read()
            print(f"Raw response body: {response_body[:100]}...")  # 디버깅용
            result = json.loads(response_body)
            
            # 결과에서 임베딩 추출
            if 'embedding' not in result:
                raise ValueError("Response does not contain 'embedding' field.")
            
            embedding = np.array(result['embedding'], dtype=np.float32)
            embeddings.append(embedding)
            print(f"Chunk {i + 1} processed successfully.")

        except Exception as e:
            print(f"Error processing chunk {i + 1}: {e}")
            continue  # 다음 청크로 진행

    print("Embeddings generation completed.")
    return embeddings


def create_faiss_index(embeddings: List[np.ndarray]) -> faiss.IndexFlatL2:
    """FAISS 인덱스를 생성."""
    print("Creating FAISS index...")
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.vstack(embeddings))
    print("FAISS index created.")
    return index

def retrieve_relevant_chunks(query: str, faiss_index: faiss.IndexFlatL2, texts: List[str], model_id: str, top_k: int = 3) -> List[str]:
    """질문과 관련된 텍스트 청크 검색."""
    print(f"Retrieving relevant chunks for query: {query}")
    query_embedding = generate_embeddings([query], model_id=model_id)[0]
    distances, indices = faiss_index.search(np.array([query_embedding]), top_k)
    print(f"Retrieved indices: {indices[0]}, distances: {distances[0]}")
    return [texts[i] for i in indices[0] if i < len(texts)]

def lambda_handler(event, context):
    print("Lambda function invoked...")
    try:
        body = json.loads(event.get('body', '{}'))
        print("Request body:", body)

        # 입력 처리
        file_content_base64 = body.get('fileContent', None)
        new_message = body.get('message', '스노우플레이크 문제 한개만 객관식으로 만들어줘')
        history = body.get('chatHistoryMessage', [])
        selectedModel = body.get('selectedModel', 'claude3.5')

        # 파일 디코딩
        file_content = None
        if file_content_base64:
            print("Decoding base64 file content...")
            print("file_content_base64 : ",file_content_base64)
            file_content = base64.b64decode(file_content_base64)

        # Bedrock 임베딩 모델 설정
        embedding_model_id = 'anthropic.claude-3-5-sonnet-20240620-v1:0'

        # Bedrock Knowledge Base 설정
        kbId = 'ZXCWNTBUPU'  # 실제 Knowledge Base ID를 입력하세요

        # 모델 ARN 설정
        if selectedModel == 'claude3.5':
            modelArn = 'arn:aws:bedrock:ap-northeast-2::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0'
        elif selectedModel == 'claude3.0':
            modelArn = 'arn:aws:bedrock:ap-northeast-2::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0'
        else:
            raise ValueError(f"Unsupported model selected: {selectedModel}")

        print(f"Selected model ARN: {modelArn}")

        # 관련 텍스트 청크 초기화
        relevant_chunks = []

        # 파일이 제공된 경우 벡터 DB 생성
        if file_content:
            print("Processing uploaded file...")
            text = extract_text_from_file(file_content)
            text_chunks = [text[i:i + 500] for i in range(0, len(text), 500)]  # 500자 단위 분할
            print(f"Generated {len(text_chunks)} text chunks.")

            # 텍스트 임베딩 생성
            embeddings = generate_embeddings(text_chunks, embedding_model_id)

            # FAISS 인덱스 생성
            faiss_index = create_faiss_index(embeddings)

            # 관련 텍스트 청크 검색
            relevant_chunks = retrieve_relevant_chunks(new_message, faiss_index, text_chunks, embedding_model_id)
            print(f"Relevant chunks: {relevant_chunks}")

        # Bedrock API 호출을 위한 프롬프트 생성
        prompt_template = {
            "system_prompt": "AWS Bedrock 모델로서 사용자의 질문에 정중하고 정확하게 답변해야 합니다.",
            "context": {"goal": "사용자의 질문에 대해 가장 정확하고 관련 있는 정보를 제공하는 것입니다."},
            "conversation": history if isinstance(history, list) else []
        }

        prompt_template["conversation"].append({"role": "user", "content": new_message})
        if relevant_chunks:
            prompt_template["context"]["retrieved_chunks"] = relevant_chunks

        print("Prompt for Bedrock API:", json.dumps(prompt_template, indent=2))

        # Bedrock API 호출
        response = bedrock_agent_runtime.retrieve_and_generate(
            input={'text': json.dumps(prompt_template)},
            retrieveAndGenerateConfiguration={
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': kbId,
                    'modelArn': modelArn
                },
                'type': 'KNOWLEDGE_BASE'
            }
        )

        # 결과 반환
        output_text = response.get('output', {}).get('text', 'No output generated')
        print("Generated response:", output_text)

        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                "responseCode": "200",
                "responseStatus": "OK",
                "resultData": {
                    "message": output_text
                }
            })
        }

    except Exception as e:
        print("Error occurred:", str(e))
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'error': str(e)
            })
        }
