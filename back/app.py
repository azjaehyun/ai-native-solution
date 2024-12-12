import asyncio
import websockets
from websockets.exceptions import ConnectionClosed

async def echo(websocket):
    client_ip = websocket.remote_address[0] if websocket.remote_address else "Unknown"
    print(f"Client connected: {client_ip}")
    try:
        async for message in websocket:
            print(f"Received from {client_ip}: {message}")
            response = f"Cluade 3.5 : {message}"
            await websocket.send(response)
    except ConnectionClosed as e:
        print(f"Connection with {client_ip} closed: {e}")
    except Exception as e:
        print(f"Unexpected error with {client_ip}: {e}")
    finally:
        print(f"Client disconnected: {client_ip}")

async def main():
    try:
        async with websockets.serve(echo, "localhost", 8765):
            print("WebSocket server started on ws://localhost:8765")
            await asyncio.Future()  # Run forever
    except Exception as e:
        print(f"Server error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped manually")
    except Exception as e:
        print(f"Fatal error on server: {e}")
