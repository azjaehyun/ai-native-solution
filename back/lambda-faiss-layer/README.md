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




## cloudshell 환경에서 아래 명령어 실행. - pdf , word , execl 계층 모듈 추가 version3.9
```
mkdir faiss_chunk
cd faiss_chunk
python3.9 -m venv venv
source venv/bin/activate
pip install openpyxl PyPDF2 python-docx  faiss-cpu 
mkdir -p python/lib/python3.9/site-packages
cp -r venv/lib/python3.9/site-packages/* python/lib/python3.9/site-packages/
cp -r venv/lib/python3.9/site-packages/.[^.]* python/lib/python3.9/site-packages/
zip -r faiss_chunk.zip python
aws lambda publish-layer-version --layer-name myPythonLayer --zip-file fileb://faiss_chunk.zip --compatible-runtimes python3.9
```

## cloudshell 환경에서 아래 명령어 실행. - pdf , word , execl 계층 모듈 추가 version 3.13
```
mkdir faiss_chunk
cd faiss_chunk
python3 -m venv venv
source venv/bin/activate
pip install openpyxl PyPDF2 python-docx
mkdir -p python/lib/python3.13/site-packages
cp -r venv/lib/python3.13/site-packages/* python/lib/python3.9/site-packages/
cp -r venv/lib/python3.13/site-packages/.[^.]* python/lib/python3.9/site-packages/
zip -r faiss_chunk.zip python
aws lambda publish-layer-version --layer-name myPythonLayer --zip-file fileb://faiss_chunk.zip --compatible-runtimes python3.13
```

# 1) 로컬 터미널에서 컨테이너 실행
docker run -it \
    -v "$PWD":/app \
    --workdir /app \
    public.ecr.aws/lambda/python:3.9 bash

# 2) 컨테이너 내부 명령어
yum install -y gcc  # (이미지에 따라 필요할 수도, 필요 없을 수도 있음)
python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install openpyxl PyPDF2 python-docx faiss-cpu

mkdir -p python/lib/python3.9/site-packages
cp -r venv/lib/python3.9/site-packages/* python/lib/python3.9/site-packages/
cp -r venv/lib/python3.9/site-packages/.[^.]* python/lib/python3.9/site-packages/ 2>/dev/null || true

zip -r faiss_chunk.zip python
exit  # 컨테이너 종료

# 3) 로컬 호스트에 faiss_chunk.zip이 생성됨. 
#    Lambda Layer로 업로드하면 됨.