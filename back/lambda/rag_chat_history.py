import boto3
import json
import uuid
from datetime import datetime

# Request body 예시
# {
#     "chatRoomId": "room123",
#     "userMessage": "오늘 날씨 어때?"
# }

# DynamoDB 클라이언트 설정
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('ChatHistory')

def lambda_handler(event, context):
    body = json.loads(event['body'])
    chat_room_id = body.get('chatRoomId')
    user_message = body.get('userMessage')

    if not chat_room_id or not user_message:
        return {"statusCode": 400, "body": "chatRoomId and userMessage are required."}

    # 이전 채팅 히스토리 가져오기
    response = table.get_item(Key={'chatRoomId': chat_room_id})
    history = response.get('Item', {}).get('history', [])

    # 새 메시지 추가
    history.append({
        "sender": "user",
        "message": user_message,
        "timestamp": datetime.utcnow().isoformat()
    })

    # RAG 기반 AWS Bedrock 호출 (가정)
    ai_response = call_bedrock(history)  # Bedrock 서비스와 연동 함수 호출

    # AI 응답 추가
    history.append({
        "sender": "bot",
        "message": ai_response,
        "timestamp": datetime.utcnow().isoformat()
    })

    # DynamoDB에 히스토리 업데이트
    table.put_item(Item={'chatRoomId': chat_room_id, 'history': history})

    return {
        "statusCode": 200,
        "body": json.dumps({
            "response": ai_response,
            "history": history
        })
    }

def call_bedrock(history):
    # Bedrock 서비스 호출 로직 추가
    return "이것은 AWS Bedrock의 예시 응답입니다."