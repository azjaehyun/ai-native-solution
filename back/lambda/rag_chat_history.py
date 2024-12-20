import boto3
import json

# Bedrock 클라이언트 생성
bedrock_agent_runtime = boto3.client(
    service_name="bedrock-agent-runtime",
    region_name="ap-northeast-2"  # 실제 사용하는 리전으로 변경
)

def map_history_to_conversation(template, history):
    # history 변수를 conversation 배열에 추가
    template["conversation"].extend(history)
    return template

def lambda_handler(event, context):
    try:
        # API Gateway로부터 전달된 입력 파라미터 처리
        body = json.loads(event.get('body', '{}'))
        
        # 새로운 메시지와 대화 히스토리 받아오기
        new_message = body.get('message', '스노우 플레이크 문제 한개만 객관식으로 만들어줘')
        history = body.get('chatHistoryMessage', [])
        selectedModel = body.get('selectedModel', 'claude3.5')

        # Bedrock Knowledge Base 설정
        kbId = 'ZXCWNTBUPU'  # 실제 Knowledge Base ID를 입력하세요

        # 모델 ARN 설정
        if selectedModel == 'claude3.5':
            modelArn = 'arn:aws:bedrock:ap-northeast-2::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0'
        elif selectedModel == 'claude3.0':
            modelArn = 'arn:aws:bedrock:ap-northeast-2::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0'
        else:
            raise ValueError(f"Unsupported model selected: {selectedModel}")

        prompt_template = {
            "system_prompt": "당신은 AWS Bedrock 모델로서 사용자의 질문에 정중하고 정확하게 답변해야 합니다.",
            "context": {
                "goal": "사용자의 질문에 대해 가장 정확하고 관련 있는 정보를 제공하는 것입니다."
            },
            "instructions": {
                "user": "질문을 명확히 하세요. 배경 정보가 있으면 제공하세요.",
                "server": "사용자의 질문에 명확하고 간결하게 답변하세요. 관련 정보가 없을 경우 솔직히 설명하세요."
            },
            "conversation": []
        }

        # 대화 히스토리를 포함한 프롬프트 생성
        updated_template = map_history_to_conversation(prompt_template, history)
        updated_template["conversation"].append({
            "role": "user",
            "content": new_message
        })
        print(json.dumps(updated_template, indent=4, ensure_ascii=False))

        # Bedrock API 호출
        response = bedrock_agent_runtime.retrieve_and_generate(
            input={'text': json.dumps(updated_template)},
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
        print(output_text)

        # Lambda의 응답 반환
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,x-api-key',
                'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                "responseCode": "200",
                "responseStatus": "OK",
                "resultData": {
                    "message": f"{output_text}"
                }
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,x-api-key',
                'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'error': str(e)
            })
        }

## clinet req param exam
# {
#     "message": "스노우플레이크 디비 성능에 관련된 문제 하나 만들어줘 객관식으로.",
#     "chatHistoryMessage": [
#         {
#             "sender": "user",
#             "text": "안녕"
#         },
#         {
#             "sender": "server",
#             "text": "안녕하세요! 어떤 질문이나 도움이 필요하신가요? 제가 도와드릴 수 있는 부분이 있다면 말씀해 주세요."
#         }
#     ]
# }

## 프롬프트 대화 
# 이 대화는 두 주체 간의 대화입니다. user는 클라이언트로서 질문을 하며, server는 AWS Bedrock 모델로서 응답합니다. server는 정중하고 정확하게 답변해야 합니다.

# user: 안녕
# server: 안녕하세요! 어떤 질문이나 도움이 필요하신가요? 제가 도와드릴 수 있는 부분이 있다면 말씀해 주세요.
# user: 이번 대화에 대한 요약을 해줘
# server: