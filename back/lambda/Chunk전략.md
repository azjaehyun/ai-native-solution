## 문단(Paragraph) 단위 청킹
하나의 문단을 하나의 Chunk로 간주합니다.
일반적으로 Word나 PDF 파일을 다룰 때, 문단 단위가 곧 의미 단위가 되는 경우가 많습니다.


##  문장(Sentence) 단위 청킹
특정 길이 이상의 문장 하나하나를 각각 하나의 Chunk로 간주합니다.
매우 긴 문단의 경우, 문장 단위로 청킹하는 것이 유용할 수 있습니다.

## 토큰(Token) 단위 청킹
토큰 길이(예: 512~1,024 토큰 등) 기준으로 나누는 방식입니다.
일반적으로 OpenAI API나 다른 LLM에서 권장하는 토큰 한계치(subset)를 넘지 않는 선에서 나눕니다.
문단이 길고 구성이 복잡할 때, 단순 문단 단위 분할보다 유연하게 대응할 수 있습니다.

## 구조(Headings, 섹션) 기반 청킹
문서가 Heading이나 섹션, 챕터 등으로 명확히 구분되어 있는 경우, 그 구조를 유지한 채로 적절한 길이를 맞춰 Chunk를 분할합니다.
예를 들어, Word의 Heading Level 1,2,3과 같은 계층 정보를 활용하거나, PDF 문서가 Outline 구조를 갖고 있다면, 해당 Outline 정보에 따라 Chunk를 나눌 수 있습니다.

## 카테고리·속성(도메인 기반) 청킹
Excel 파일 등 표 형식 데이터를 다룰 때는, 시트별·열(Column)별·구간별로 의미 있는 단위를 찾아서 청킹할 수 있습니다(예: '고객 정보' 시트, '매출' 시트, 특정 기간 등).
텍스트보다는 구조화된 데이터가 많으므로, “행(row)” 단위 혹은 “특정 칼럼 조합” 단위로 나눌 수도 있습니다.



## 문단(Paragraph) 단위 청킹
## 문장(Sentence) 단위 청킹
## 토큰(Token) 단위 청킹
## 구조(Headings, 섹션) 기반 청킹
## 카테고리·속성(도메인 기반) 청킹



# rag_chat_fileupload_faiss_chunk_local.py 사용 가이드 
```
pip install boto3 faiss-cpu numpy python-docx PyPDF2 openpyxl
aws configure
python local_chunk_test.py \
  --file "/path/to/document.pdf" \
  --chunkStrategy "sentence" \
  --maxChunkSize 400

python local_chunk_test.py \
  --file "/path/to/document.pdf" \
  --chunkStrategy "sentence" \
  --maxChunkSize 400 \
  --overlapSize 50 \
  --useEmbedding \
  --query "이 문서의 핵심 요약은 무엇인가?"


python local_chunk_test.py \
  --file "/path/to/long_text.txt" \
  --chunkStrategy "token" \
  --maxChunkSize 100 \
  --overlapSize 10
```