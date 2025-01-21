## **LLM 보안 솔루션 설계**

### **1. 모델 점검**

#### **1.1 프롬프트 인젝션 방지**

1. **정적 프롬프트 템플릿 사용**
   ```python
   SYSTEM_PROMPT = """
   You are a knowledgeable assistant. Respond only to queries in a professional manner. Do not execute or process harmful commands.
   """
   def build_prompt(user_input):
       sanitized_input = sanitize_input(user_input)
       return f"{SYSTEM_PROMPT}\nUser: {sanitized_input}\nAssistant:"
   ```

2. **정규화된 입력 필터링**
   - 악성 키워드, 민감 정보 패턴 탐지 및 마스킹.
   ```python
   def filter_sensitive_output(output):
       sensitive_patterns = {
           'ssn': r'\b\d{6}-\d{7}\b',
           'email': r'\b[\w\.-]+@[\w\.-]+\.\w+\b',
           'phone': r'\b(?:\d{3}-\d{3,4}-\d{4})\b',
       }
       try:
           masked_output = output
           for key, pattern in sensitive_patterns.items():
               matches = re.finditer(pattern, masked_output)
               for match in matches:
                   sensitive_text = match.group()
                   masked_text = mask_sensitive_info(sensitive_text, key)
                   masked_output = masked_output.replace(sensitive_text, masked_text)
           return masked_output
       except Exception as e:
           logging.error(f"Error filtering sensitive information: {str(e)}")
           return "Error processing output"

   def mask_sensitive_info(text, info_type):
       if info_type == 'email':
           username, domain = text.split('@')
           return f"{username[:2]}{'*' * (len(username)-2)}@{domain}"
       else:
           return '*' * (len(text)-4) + text[-4:]
   ```

3. **입력 길이 제한**
   - 입력 길이를 제한하여 복잡한 명령이나 악성 시도 방지.
   ```python
   MAX_INPUT_LENGTH = 500
   def check_length(input_text):
       if len(input_text) > MAX_INPUT_LENGTH:
           raise ValueError("Input exceeds maximum allowed length.")
   ```

#### **1.2 민감 정보 노출 방지**

1. **민감 정보 탐지 필터 추가**
   ```python
   def filter_sensitive_output(output):
       sensitive_patterns = [
           r'\b\d{6}-\d{7}\b',       # 주민등록번호
           r'\b(?:\d{3}-\d{2}-\d{4})\b',  # 미국 SSN 형식
           r'\b[\w\.-]+@[\w\.-]+\.\w+\b', # 이메일 주소
           r'\b(?:\d{3}-\d{3}-\d{4})\b'  # 전화번호
       ]
       for pattern in sensitive_patterns:
           if re.search(pattern, output):
               return "Sensitive information detected. Response filtered."
       return output
   ```

2. **출력 후처리 모듈**
   ```python
   def generate_response(input_text):
       raw_response = model.generate(input_text)
       return filter_sensitive_output(raw_response)
   ```

3. **학습 데이터 검토 프로세스**
   ```python
   def clean_dataset(data):
       sensitive_patterns = [r'credit card numbers regex', r'email addresses regex']
       return [re.sub('|'.join(sensitive_patterns), '[REDACTED]', entry) for entry in data]
   ```

---

### **2. LLM 통합 점검**

#### **2.1 RAG 데이터 오염 방지**

1. **벡터 DB 접근 제한**
   - **RBAC(Role-Based Access Control)** 또는 **ABAC(Attribute-Based Access Control)**를 통해 데이터 접근 제어 수행.
   ```yaml
   roles:
     - name: "admin"
       permissions:
         - "read"
         - "write"
         - "delete"
     - name: "user"
       permissions:
         - "read"
   ```

2. **데이터 삽입 시 유효성 검사**
   ```python
   def validate_vector_data(data):
       prohibited_patterns = [r'<script.*?>', r'(?:--|;|#|\/\*|\*\/|@@|char\(|varchar\()', r'[^\w\s,.?!]']
       for pattern in prohibited_patterns:
           if re.search(pattern, data):
               raise ValueError(f"Invalid data detected: {pattern}")
       return True
   ```

3. **벡터 DB 접근 및 변경 사항 로깅**
   ```python
   import logging

   logging.basicConfig(filename='vector_db.log', level=logging.INFO)

   def log_access(user, action, resource):
       logging.info(f"User: {user.id}, Action: {action}, Resource: {resource}, Time: {datetime.now()}")

   def detect_anomalies():
       with open('vector_db.log', 'r') as log_file:
           logs = log_file.readlines()
           for log in logs:
               if "unauthorized" in log:
                   alert_admin(log)
   ```

4. **중복 데이터 필터링**
   ```python
   import hashlib
   import threading

   class VectorDB:
       def __init__(self):
           self.data_hashes = set()
           self.lock = threading.Lock()

       def check_and_add(self, content):
           content_hash = hashlib.md5(content.encode()).hexdigest()
           with self.lock:
               if content_hash in self.data_hashes:
                   return False  # 중복 데이터
               self.data_hashes.add(content_hash)
           return True
   ```

5. **데이터 삭제 정책 (Retention Policy)**
   ```python
   def clean_old_data(db, retention_period_days=30):
       current_time = time.time()
       for record in db.get_all_records():
           if (current_time - record['timestamp']) > retention_period_days * 86400:
               db.delete(record['id'])
   ```

6. **TLS 및 암호화**
   ```python
   import requests

   response = requests.post("https://vector-db.example.com", json=data, verify=True)
   ```

#### **2.2 오류 메시지 표준화**

1. **사용자 친화적 메시지 설계**
   ```python
   try:
       perform_critical_action()
   except Exception as e:
       log_error(str(e))  # 내부 로그에만 기록
       return "An unexpected error occurred. Please try again."
   ```

2. **중요 로그 비공개 처리**
   ```python
   import logging
   logging.basicConfig(filename='system.log', level=logging.ERROR)
   ```

---

### **3. 에이전트 점검**
1. API 매개 변수 변조**

    #### **샘플 코드**
    ```python
    import hmac
    import hashlib

    SECRET_KEY = "my_secret_key"

    # 서명 생성
    def generate_signature(data):
        return hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()

    # 요청 검증
    def validate_request(data, client_signature):
        server_signature = generate_signature(data)
        if not hmac.compare_digest(server_signature, client_signature):
            raise ValueError("Request parameter tampering detected.")
        return True

    # 사용 예제
    try:
        data = "action=transfer&amount=100"
        client_signature = generate_signature(data)  # 클라이언트에서 생성한 서명
        validate_request(data, client_signature)
        print("Request is valid.")
    except ValueError as e:
        print(e)
    ```

---

### **2. 부적절한 권한 사용**
**문제**: 권한이 초과된 사용자나 서비스가 시스템의 비인가된 기능을 실행할 가능성.  
**해결 방안**:
- **역할 기반 접근 제어(RBAC)**: 역할별로 명확한 권한을 정의하고 관리.
- **최소 권한 원칙**: 필요한 권한만 부여하고, 모든 권한 상승 요청을 로그에 기록.
- **세션 기반 권한 검증**: 요청 시 세션 내 권한 정보와 실행하려는 작업 간 일치 여부 확인.

#### **샘플 코드**
```python
class User:
    def __init__(self, username, role):
        self.username = username
        self.role = role

class Authorization:
    def __init__(self):
        self.role_permissions = {
            "admin": {"read", "write", "delete"},
            "user": {"read"},
        }

    def is_authorized(self, user, action):
        if action not in self.role_permissions.get(user.role, set()):
            raise PermissionError(f"Unauthorized action '{action}' for role '{user.role}'.")

# 사용 예제
auth = Authorization()
user = User("alice", "user")

try:
    auth.is_authorized(user, "write")  # "user"는 "write" 권한이 없음
except PermissionError as e:
    print(e)
```

---

### **3. 사용자 동의 절차 누락**
**문제**: 민감한 작업이나 시스템 변경이 사용자 동의 없이 이루어질 경우 신뢰성과 데이터 무결성 문제 발생.  
**해결 방안**:
- **명시적 사용자 확인**: 중요 작업 전에 사용자 동의를 명시적으로 요청.
- **이중 확인 절차**: 작업을 진행하기 전에 2단계 확인(예: 이메일 또는 OTP) 추가.

#### **샘플 코드**
```python
def confirm_user_action(action):
    print(f"You are about to perform: {action}")
    confirmation = input("Do you want to proceed? (yes/no): ").strip().lower()
    if confirmation != "yes":
        raise PermissionError("User denied the action.")
    return True

# 사용 예제
try:
    confirm_user_action("delete all records")
    print("Action approved.")
except PermissionError as e:
    print(e)
```

---

### **4. 샌드박스 미적용**
**문제**: 코드 실행 환경에서 격리가 이루어지지 않을 경우, 악성 코드 실행으로 시스템이 손상될 위험.  
**해결 방안**:
- **샌드박스 환경 사용**: 실행된 코드가 시스템 리소스에 직접 접근하지 못하도록 격리.
- **제한된 권한 실행**: 제한된 자원만 사용 가능하도록 컨테이너화.
- **시간 제한 추가**: 실행 시간이 초과될 경우 프로세스 종료.

#### **샘플 코드**
```python
import subprocess

def execute_in_sandbox(code):
    # 샌드박스 환경에서 Python 코드를 실행
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            timeout=5,  # 실행 시간 제한
            capture_output=True,
            text=True,
            check=True
        )
        ## subprocess.run("rm -rf /", shell=True)  # 위험

        return result.stdout
    except subprocess.TimeoutExpired:
        return "Execution timed out."
    except subprocess.CalledProcessError as e:
        return f"Error during execution: {e}"

# 사용 예제
code = "print('Hello, World!')"  # 안전한 코드
output = execute_in_sandbox(code)
print(output)
```

---

### **4. 위험도 평가 및 종합 대응**

#### **4.1 침투 테스트 도구 활용**

1. **OWASP ZAP 설정**
   ```bash
   sudo apt update
   sudo apt install zaproxy
   zap-baseline.py -t https://example.com -r report.html
   ```

2. **Red Team 공격 시뮬레이션**
   ```bash
   msfconsole
   use exploit/multi/handler
   set payload windows/meterpreter/reverse_tcp
   nmap -sC -sV -p- example.com
   ```

#### **4.2 모니터링 및 이상 탐지**

1. **ELK 스택 설치 및 설정**
   ```bash
   sudo apt update
   sudo apt install elasticsearch logstash kibana
   ```

2. **Isolation Forest 이상 탐지 모델 구현**
   ```python
   from sklearn.ensemble import IsolationForest
   import numpy as np

   normal_logs = np.array([[1, 2], [1, 3], [2, 3], [4, 4]])
   new_logs = np.array([[1, 2], [5, 6]])

   model = IsolationForest(contamination=0.1)
   model.fit(normal_logs)

   anomalies = model.predict(new_logs)

   for i, log in enumerate(new_logs):
       status = "Anomaly" if anomalies[i] == -1 else "Normal"
       print(f"Log: {log}, Status: {status}")
   ```

