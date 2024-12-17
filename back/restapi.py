from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # CORS 설정

@app.route('/echo', methods=['POST', 'GET', 'PUT', 'DELETE'])
def echo():
    try:
        print(request)
        if request.is_json:
            data = request.json
        else:
            data = request.form.to_dict() or request.args.to_dict()

        message = data.get('message', '')
        response = {
            "responseCode": "200",
            "responseStatus": "OK",
            "resultData": {
                "message": f"Claude 3.5 : {message}"
            }
        }
        return jsonify(response)
    except Exception as e:
        error_response = {
            "responseCode": "500",
            "responseStatus": "Error",
            "resultData": {
                "error": str(e)
            }
        }
        return jsonify(error_response), 500

if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0', port=8765)
    except KeyboardInterrupt:
        print("Server stopped manually")
    except Exception as e:
        print(f"Fatal error on server: {e}")