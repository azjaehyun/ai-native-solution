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
    return file_content.decode('utf-8')  # 간단히 디코딩

def generate_embeddings(text: List[str], model_id: str) -> List[np.ndarray]:
    """Amazon Bedrock을 사용하여 텍스트 임베딩 생성."""
    bedrock_runtime = boto3.client('bedrock-runtime', region_name='ap-northeast-2')
    embeddings = []

    for chunk in text:
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            contentType='application/json',
            accept='application/json',
            body=json.dumps({"input": chunk})
        )
        result = json.loads(response['body'].read())
        embeddings.append(np.array(result['embedding'], dtype=np.float32))

    return embeddings

def create_faiss_index(embeddings: List[np.ndarray]) -> faiss.IndexFlatL2:
    """FAISS 인덱스를 생성."""
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.vstack(embeddings))
    return index

def retrieve_relevant_chunks(query: str, faiss_index: faiss.IndexFlatL2, texts: List[str], model_id: str, top_k: int = 3) -> List[str]:
    """질문과 관련된 텍스트 청크 검색."""
    query_embedding = generate_embeddings([query], model_id=model_id)[0]
    distances, indices = faiss_index.search(np.array([query_embedding]), top_k)
    return [texts[i] for i in indices[0] if i < len(texts)]

def lambda_handler(event, context):
    try:
        body = json.loads(event.get('body', '{}'))

        # 입력 처리
        file_content_base64 = body.get('fileContent', None)
        new_message = body.get('message', '스노우플레이크 문제 한개만 객관식으로 만들어줘')
        history = body.get('chatHistoryMessage', [])
        selectedModel = body.get('selectedModel', 'claude3.5')

        # 파일 디코딩
        file_content = None
        if file_content_base64:
            file_content = base64.b64decode(file_content_base64)

        # Bedrock 임베딩 모델 설정
        embedding_model_id = 'amazon.titan-embed-text-v1'

        # 모델 ARN 설정
        if selectedModel == 'claude3.5':
            modelArn = 'arn:aws:bedrock:ap-northeast-2::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0'
        elif selectedModel == 'claude3.0':
            modelArn = 'arn:aws:bedrock:ap-northeast-2::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0'
        else:
            raise ValueError(f"Unsupported model selected: {selectedModel}")

        # 관련 텍스트 청크 초기화
        relevant_chunks = []

        # 파일이 제공된 경우 벡터 DB 생성
        if file_content:
            text = extract_text_from_file(file_content)
            text_chunks = [text[i:i + 500] for i in range(0, len(text), 500)]  # 500자 단위 분할

            # 텍스트 임베딩 생성
            embeddings = generate_embeddings(text_chunks, embedding_model_id)

            # FAISS 인덱스 생성
            faiss_index = create_faiss_index(embeddings)

            # 관련 텍스트 청크 검색
            relevant_chunks = retrieve_relevant_chunks(new_message, faiss_index, text_chunks, embedding_model_id)

        # Bedrock API 호출을 위한 프롬프트 생성
        prompt_template = {
            "system_prompt": "AWS Bedrock 모델로서 사용자의 질문에 정중하고 정확하게 답변해야 합니다.",
            "context": {"goal": "사용자의 질문에 대해 가장 정확하고 관련 있는 정보를 제공하는 것입니다."},
            "conversation": history if isinstance(history, list) else []
        }

        prompt_template["conversation"].append({"role": "user", "content": new_message})
        if relevant_chunks:
            prompt_template["context"]["retrieved_chunks"] = relevant_chunks

        # Bedrock API 호출
        response = bedrock_agent_runtime.retrieve_and_generate(
            input={'text': json.dumps(prompt_template)},
            retrieveAndGenerateConfiguration={
                'knowledgeBaseConfiguration': {
                    'modelArn': modelArn
                },
                'type': 'KNOWLEDGE_BASE'
            }
        )

        # 결과 반환
        output_text = response.get('output', {}).get('text', 'No output generated')
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