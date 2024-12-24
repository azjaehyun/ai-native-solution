## cloudshell 환경에서 아래 명령어 실행.
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