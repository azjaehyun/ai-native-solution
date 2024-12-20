`penv`를 사용하여 Lambda에서 필요한 라이브러리(`numpy`와 `faiss`)를 설치하고 계층(layer)을 생성하는 방법은 다음과 같습니다:

---

### 1. `penv` 설치
먼저, `penv`를 설치합니다. `penv`는 Python 패키지를 간단히 Lambda 계층으로 변환해주는 도구입니다.

```bash
pip install penv
```

---

### 2. `penv`로 계층 생성
`penv`를 사용하여 필요한 패키지를 설치하고 계층 파일을 생성합니다.

```bash
# faiss와 numpy를 포함한 계층 생성
penv create faiss-numpy-layer --packages numpy faiss-cpu
```

---

### 3. 계층 압축
`penv`가 생성한 `faiss-numpy-layer` 디렉터리를 Lambda 계층으로 업로드할 수 있도록 압축합니다.

```bash
cd faiss-numpy-layer
zip -r faiss-numpy-layer.zip .
cd ..
```

---

### 4. AWS Lambda 계층 생성
AWS CLI를 사용하여 Lambda 계층을 생성합니다.

```bash
aws lambda publish-layer-version \
    --layer-name faiss-numpy-layer \
    --description "FAISS and numpy for vector search" \
    --zip-file fileb://faiss-numpy-layer.zip \
    --compatible-runtimes python3.8 python3.9 python3.10 \
    --region ap-northeast-2
```

---

### 5. Lambda 함수에 계층 추가
생성된 계층을 Lambda 함수에 추가합니다. `--layers` 값은 계층 생성 결과의 ARN으로 대체해야 합니다.

```bash
aws lambda update-function-configuration \
    --function-name your-lambda-function-name \
    --layers arn:aws:lambda:ap-northeast-2:123456789012:layer:faiss-numpy-layer:1
```

---

### 6. Lambda 테스트
Lambda 콘솔이나 AWS CLI를 사용하여 함수를 호출하여 올바르게 작동하는지 확인합니다.

```bash
aws lambda invoke \
    --function-name your-lambda-function-name \
    --payload '{}' \
    response.json
```

---

이렇게 하면 `penv`를 사용하여 `numpy`와 `faiss`를 포함한 계층을 쉽게 생성하고, Lambda에서 사용할 수 있습니다. 추가적인 도움이 필요하면 알려주세요!