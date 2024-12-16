import asyncio
import websockets
import json

# API Gateway에서 제공하는 WebSocket URL을 사용 (예: wss://your-api-id.execute-api.region.amazonaws.com/your-stage)
API_GATEWAY_URL = "wss://wxyl8vbopa.execute-api.us-east-1.amazonaws.com/production"

async def connect_to_websocket():
    # WebSocket 서버에 연결
    async with websockets.connect(API_GATEWAY_URL) as websocket:
        print("Connected to WebSocket")

        # 서버로 메시지 보내기
        message = {"message": "Hello, server!"}
        await websocket.send(json.dumps(message))
        print("Message sent to server")

        # 서버로부터 응답 받기
        response = await websocket.recv()
        print(f"Message from server: {response}")

# 이미 실행 중인 이벤트 루프를 가져와서 비동기 함수 실행
loop = asyncio.get_event_loop()
loop.run_until_complete(connect_to_websocket())