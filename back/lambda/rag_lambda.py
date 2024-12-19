import boto3
import json

# Bedrock 클라이언트 생성
bedrock_agent_runtime = boto3.client(
    service_name="bedrock-agent-runtime",
    region_name="us-east-1"  # 실제 사용하는 리전으로 변경
)

# test param
# {
#   "input": "미움을 파는 고슴도치 동화책으로 10문제 만들어줄래?",
#   "question_count": "10",
#   "difficulty_level": "쉽게"
# }

# Lambda 핸들러 함수
def lambda_handler(event, context):
    try:
        # API Gateway로부터 전달된 입력 파라미터 처리
        body = json.loads(event.get('body', '{}'))
        input_text = body.get('message', '기본 입력 텍스트입니다.')

        # Bedrock Knowledge Base 설정
        kbId = 'YOUR_KNOWLEDGE_BASE_ID'  # 여기에 실제 Knowledge Base ID를 입력하세요
        modelArn = 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0'

        # Bedrock API 호출
        response = bedrock_agent_runtime.retrieve_and_generate(
            input={'text': input_text},
            retrieveAndGenerateConfiguration={
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': kbId,
                    'modelArn': modelArn
                },
                'type': 'KNOWLEDGE_BASE'
            }
        )

        # Bedrock API 결과 처리
        output_text = response.get('output', {}).get('text', 'No output generated')

        # Lambda의 응답 반환
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'message': "RAG 기반 응답이 성공적으로 생성되었습니다.",
                'output': output_text
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