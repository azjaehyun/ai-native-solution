## cloudshell 환경에서 아래 명령어 실행. - faiss 계층 모듈 추가
```
mkdir -p lambda-layer && cd lambda-layer
sudo yum update -y
mkdir -p python/lib/python3.9/site-packages
# FAISS 설치 (CPU 버전)
pip3 install faiss-cpu -t python/lib/python3.9/site-packages/
# 필요한 공유 라이브러리 복사
cp /usr/lib64/libgomp.so.1 python/lib/
zip -r faiss-layer.zip python
```




## cloudshell 환경에서 아래 명령어 실행. - pdf , word , execl 계층 모듈 추가
```
mkdir faiss_chunk
cd faiss_chunk
python3.9 -m venv venv
source venv/bin/activate
pip install openpyxl PyPDF2 python-docx faiss-cpu
mkdir -p python/lib/python3.9/site-packages
cp -r venv/lib/python3.9/site-packages/* python/lib/python3.9/site-packages/
cp -r venv/lib/python3.9/site-packages/.[^.]* python/lib/python3.9/site-packages/
zip -r faiss_chunk.zip python
aws lambda publish-layer-version --layer-name myPythonLayer --zip-file fileb://faiss_chunk.zip --compatible-runtimes python3.9
```


## docker 환경에서의 lambda layer 생성

`zip` 유틸리티가 Docker 컨테이너에 설치되어 있지 않아서 발생하는 문제입니다. 이를 해결하기 위해 Docker 컨테이너 내부에서 `zip`을 설치한 후 다시 시도해야 합니다. 아래는 전체 과정을 단계별로 안내합니다.

## 1. Docker 컨테이너 실행

먼저, Docker 컨테이너를 올바른 아키텍처와 설정으로 실행합니다. 현재 사용 중인 명령어는 올바르지만, `zip` 유틸리티가 없기 때문에 추가 설치가 필요합니다.

```bash
docker run --platform=linux/amd64 -it --rm \
    --entrypoint /bin/bash \
    -v "$PWD":/app \
    --workdir /app \
    public.ecr.aws/lambda/python:3.9
```

## 2. 컨테이너 내부에서 필요한 도구 설치 및 패키지 설정

컨테이너가 실행되고 `bash` 셸에 들어가면, 다음 명령어를 순서대로 실행합니다.

### (1) 시스템 패키지 업데이트 및 `zip` 설치

`zip` 유틸리티가 없기 때문에 먼저 `yum`을 사용하여 이를 설치합니다.

```bash
yum update -y
yum install -y gcc zip
```

- **`yum update -y`**: 시스템 패키지를 최신 상태로 업데이트합니다.
- **`yum install -y gcc zip`**: `gcc` 컴파일러와 `zip` 유틸리티를 설치합니다.

### (2) 가상 환경 생성 및 활성화

Python 가상 환경을 생성하고 활성화합니다.

```bash
python3.9 -m venv venv
source venv/bin/activate
```

### (3) `pip` 업그레이드 및 필요한 라이브러리 설치

`pip`을 최신 버전으로 업그레이드한 후, 필요한 Python 패키지를 설치합니다.

```bash
pip install --upgrade pip
pip install openpyxl PyPDF2 python-docx faiss-cpu
```

### (4) Lambda Layer 구조 생성

Lambda Layer는 특정 디렉토리 구조를 가져야 합니다. 이를 위해 필요한 디렉토리를 생성합니다.

```bash
mkdir -p python/lib/python3.9/site-packages
```

### (5) `site-packages` 복사

가상 환경의 `site-packages` 디렉토리 내용을 Lambda Layer 구조로 복사합니다.

```bash
cp -r venv/lib/python3.9/site-packages/* python/lib/python3.9/site-packages/
cp -r venv/lib/python3.9/site-packages/.[^.]* python/lib/python3.9/site-packages/ 2>/dev/null || true
```

- **`2>/dev/null || true`**: 숨김 파일 복사 중 발생할 수 있는 에러 메시지를 무시합니다.

### (6) `zip`으로 Layer 패키징

이제 `python` 디렉토리를 `zip` 파일으로 압축합니다.

```bash
zip -r faiss_chunk.zip python
```

### (7) 컨테이너 종료

패키징이 완료되면 컨테이너를 종료합니다.

```bash
exit
```

## 3. 로컬 호스트에서 `faiss_chunk.zip` 확인 및 Lambda Layer 업로드

컨테이너를 종료하면, 호스트 머신(맥 M2)의 현재 디렉토리에 `faiss_chunk.zip` 파일이 생성됩니다. 이 파일을 AWS Lambda Layer로 업로드할 수 있습니다.

## 전체 과정 요약

아래는 전체 과정을 한 번에 정리한 명령어입니다. 각 단계별로 설명과 함께 진행하시면 됩니다.

### 1) Docker 컨테이너 실행

```bash
docker run --platform=linux/amd64 -it --rm \
    --entrypoint /bin/bash \
    -v "$PWD":/app \
    --workdir /app \
    public.ecr.aws/lambda/python:3.9
```

### 2) 컨테이너 내부 명령어 실행

```bash
# 시스템 업데이트 및 zip 설치
yum update -y
yum install -y gcc zip

# 가상 환경 생성 및 활성화
python3.9 -m venv venv
source venv/bin/activate

# pip 업그레이드 및 라이브러리 설치
pip install --upgrade pip
pip install openpyxl PyPDF2 python-docx faiss-cpu

# Lambda Layer 디렉토리 구조 생성
mkdir -p python/lib/python3.9/site-packages

# site-packages 복사
cp -r venv/lib/python3.9/site-packages/* python/lib/python3.9/site-packages/
cp -r venv/lib/python3.9/site-packages/.[^.]* python/lib/python3.9/site-packages/ 2>/dev/null || true

# Layer 패키징
zip -r faiss_chunk.zip python

# 컨테이너 종료
exit
```

### 3) 로컬에서 Lambda Layer 업로드

호스트 머신의 현재 디렉토리에 생성된 `faiss_chunk.zip` 파일을 AWS Lambda 콘솔을 통해 Layer로 업로드합니다.

## 추가 팁

- **Docker 이미지 선택**: 만약 추가적으로 필요한 패키지가 있다면, Amazon Linux 2 또는 SAM 빌드 이미지를 사용하는 것도 고려해볼 수 있습니다.
- **캐싱 활용**: 자주 사용하는 패키지는 Docker 이미지에 미리 설치해두면 빌드 시간을 단축할 수 있습니다.
- **디버깅**: 만약 다른 패키지에서도 문제가 발생한다면, 필요한 시스템 패키지를 추가로 설치해야 할 수 있습니다. 예를 들어, `libxml2-devel` 같은 패키지가 필요할 수 있습니다.

## 참고 자료

- [AWS Lambda Layers 공식 문서](https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html)
- [AWS 공식 ECR Lambda 이미지](https://gallery.ecr.aws/lambda)
- [Amazon Linux 2 Docker 이미지](https://hub.docker.com/_/amazonlinux)

이 과정을 통해 Lambda Layer를 성공적으로 생성하고, `GLIBC` 관련 오류 없이 Python 패키지를 사용할 수 있을 것입니다. 추가적인 문제가 발생하면 언제든지 질문해 주세요!

