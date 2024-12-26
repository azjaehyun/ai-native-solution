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
pip install openpyxl PyPDF2 python-docx
mkdir -p python/lib/python3.9/site-packages
cp -r venv/lib/python3.9/site-packages/* python/lib/python3.9/site-packages/
cp -r venv/lib/python3.9/site-packages/.[^.]* python/lib/python3.9/site-packages/
zip -r faiss_chunk.zip python
aws lambda publish-layer-version --layer-name myPythonLayer --zip-file fileb://faiss_chunk.zip --compatible-runtimes python3.9
```

