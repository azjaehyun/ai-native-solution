import boto3
import json
import faiss
import numpy as np
import base64
from typing import List
from io import BytesIO

# (1) Word 파일 처리용 라이브러리
try:
    from docx import Document
except ImportError:
    pass

# (2) PDF 파일 처리용 라이브러리
try:
    import PyPDF2
except ImportError:
    pass

# (3) Excel 파일 처리용 라이브러리
try:
    from openpyxl import load_workbook
except ImportError:
    pass

# Bedrock 클라이언트 생성
bedrock_agent_runtime = boto3.client(
    service_name="bedrock-agent-runtime",
    region_name="ap-northeast-2"  # 실제 사용하는 리전으로 변경
)


def dynamic_chunk_text(
    text: str,
    strategy: str = "fixed",
    chunk_size: int = 500,
    overlap_size: int = 50
) -> List[str]:
    """
    일반 텍스트용 동적 청킹 함수.
    
    strategy:
      - "paragraph": \n\n 기반 문단 분리 후, 길면 슬라이딩 윈도우
      - "sentence": 마침표/물음표/느낌표를 기준으로 문장 분리 후, 길면 슬라이딩 윈도우
      - "fixed": 일정 크기(chunk_size)로 고정 분할
      - "token": (간단히) 공백 기반 단어 개수를 세어 chunk_size 단위로 분할 (Naive)
    """
    text = text.strip()
    if not text:
        return []

    if strategy == "paragraph":
        paragraphs = text.split('\n\n')
        chunks = []
        for para in paragraphs:
            para = para.strip()
            if len(para) <= chunk_size:
                if para:
                    chunks.append(para)
            else:
                # 슬라이딩 윈도우 (문자 길이 기준)
                start = 0
                while start < len(para):
                    end = start + chunk_size
                    chunk = para[start:end]
                    chunks.append(chunk)
                    start += max(chunk_size - overlap_size, 1)
        return chunks

    elif strategy == "sentence":
        import re
        # 마침표/물음표/느낌표 뒤 공백/줄바꿈을 문장 구분자로 가정
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) <= chunk_size:
                if sent:
                    chunks.append(sent)
            else:
                start = 0
                while start < len(sent):
                    end = start + chunk_size
                    chunk = sent[start:end]
                    chunks.append(chunk)
                    start += max(chunk_size - overlap_size, 1)
        return chunks

    elif strategy == "token":
        # 여기서는 간단히 '공백' 기준으로 단어를 나눈 뒤,
        # chunk_size 개수만큼씩 묶어서 하나의 청크로 만든 예시
        words = text.split()
        chunks = []
        start_idx = 0
        while start_idx < len(words):
            end_idx = start_idx + chunk_size
            chunk_words = words[start_idx:end_idx]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
            start_idx += max(chunk_size - overlap_size, 1)
        return chunks

    else:
        # "fixed" (문자 기반 고정 길이)
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size
        return chunks


def chunk_word(
    file_bytes: bytes,
    chunk_size: int = 500,
    overlap_size: int = 50
) -> List[str]:
    """
    Word(.docx) 파일을 paragraph 단위로 읽어서,
    각 문단이 너무 길면 슬라이딩 윈도우로 분할.
    """
    chunks = []
    try:
        doc = Document(BytesIO(file_bytes))
    except Exception as e:
        print(f"Error loading Word docx: {e}")
        return []

    for i, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if not text:
            continue

        if len(text) <= chunk_size:
            chunks.append(f"[Paragraph {i+1}] {text}")
        else:
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk = text[start:end]
                chunks.append(f"[Paragraph {i+1}] {chunk}")
                start += max(chunk_size - overlap_size, 1)

    return chunks

def chunk_word_heading(
    file_bytes: bytes,
    chunk_size: int = 500,
    overlap_size: int = 50
) -> List[str]:
    """
    Word(.docx) 파일에서 Heading(헤딩) 스타일을 기준으로 분할하는 예시.
    - Heading 스타일 감지: paragraph.style.name 안에 'Heading'이 있는지 확인 (Naive)
    - Heading에서 Heading까지를 하나의 청크로 묶음
    - 각 청크가 너무 길면 슬라이딩 윈도우로 분할
    """
    chunks = []
    try:
        doc = Document(BytesIO(file_bytes))
    except Exception as e:
        print(f"Error loading Word docx: {e}")
        return []

    all_paragraphs = doc.paragraphs
    current_heading = None
    current_text_buffer = []

    def flush_chunk(heading, text_buffer):
        """
        내부 함수: heading + buffer를 하나의 chunk로 만들고,
        chunk_size 초과 시 슬라이딩 윈도우
        """
        if not text_buffer:
            return []
        combined_text = f"[Heading: {heading}] " + "\n".join(text_buffer)
        if len(combined_text) <= chunk_size:
            return [combined_text]
        else:
            splitted_chunks = []
            start = 0
            while start < len(combined_text):
                end = start + chunk_size
                chunk_part = combined_text[start:end]
                splitted_chunks.append(chunk_part)
                start += max(chunk_size - overlap_size, 1)
            return splitted_chunks

    for para in all_paragraphs:
        style_name = getattr(para.style, 'name', '') or ''
        text = para.text.strip()
        # Heading 스타일 감지
        if 'Heading' in style_name:
            # 이전 버퍼가 있으면 먼저 flush
            if current_heading or current_text_buffer:
                chunks.extend(flush_chunk(current_heading, current_text_buffer))
            # 새 Heading 시작
            current_heading = text
            current_text_buffer = []
        else:
            if not current_heading:
                # Heading 없이 시작된 문단은 "Unknown Heading" 처리
                current_heading = "No Heading"
            # 현재 Heading 하에 문단 추가
            if text:
                current_text_buffer.append(text)

    # 마지막 버퍼 flush
    if current_heading or current_text_buffer:
        chunks.extend(flush_chunk(current_heading, current_text_buffer))

    return chunks


def chunk_pdf(
    file_bytes: bytes,
    chunk_size: int = 500,
    overlap_size: int = 50
) -> List[str]:
    """
    PDF 파일을 페이지 단위로 텍스트 추출 후,
    페이지 텍스트가 너무 길면 슬라이딩 윈도우로 분할.
    """
    chunks = []
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []

    for page_index, page in enumerate(pdf_reader.pages):
        try:
            text = page.extract_text()
        except:
            text = ""
        text = text.strip() if text else ""
        if not text:
            continue

        if len(text) <= chunk_size:
            chunks.append(f"[Page {page_index+1}] {text}")
        else:
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk = text[start:end]
                chunks.append(f"[Page {page_index+1}] {chunk}")
                start += max(chunk_size - overlap_size, 1)

    return chunks


def chunk_excel(
    file_bytes: bytes,
    chunk_size: int = 500,
    overlap_size: int = 50
) -> List[str]:
    """
    Excel 파일을 시트(탭) → 행(Row) 순으로 순회 후,
    각 행을 문자열로 연결해 한 덩어리로 만들고,
    너무 길면 슬라이딩 윈도우(chunk_size/overlap_size)로 분할.
    """
    chunks = []
    try:
        wb = load_workbook(filename=BytesIO(file_bytes), read_only=True, data_only=True)
    except Exception as e:
        print(f"Error loading Excel: {e}")
        return []

    for sheet_name in wb.sheetnames:
        sheet = wb[sheet_name]
        print(f"Processing sheet: {sheet_name}")
        for row in sheet.iter_rows(values_only=True):
            row_str_list = [str(cell) if cell is not None else '' for cell in row]
            row_text = " | ".join(row_str_list).strip()

            if not row_text:
                continue

            if len(row_text) <= chunk_size:
                chunks.append(f"[Sheet: {sheet_name}] {row_text}")
            else:
                start = 0
                while start < len(row_text):
                    end = start + chunk_size
                    chunk = row_text[start:end]
                    chunks.append(f"[Sheet: {sheet_name}] {chunk}")
                    start += max(chunk_size - overlap_size, 1)
    wb.close()
    return chunks


def chunk_excel_domain_based(
    file_bytes: bytes,
    category_col: int = 0
) -> List[str]:
    """
    Excel에서 '카테고리·속성(도메인 기반)'으로 묶는 간단 예시.
    
    - category_col(0-based): 특정 열이 "카테고리"라고 가정
    - 동일 카테고리 값인 행들을 모아서 한 덩어리로 만든 뒤, return
    
    (문자/토큰 길이 기반 슬라이딩 윈도우는 생략, 도메인별로 chunk를 합침)
    """
    chunks = []
    try:
        wb = load_workbook(filename=BytesIO(file_bytes), read_only=True, data_only=True)
    except Exception as e:
        print(f"Error loading Excel for domain chunking: {e}")
        return []

    for sheet_name in wb.sheetnames:
        sheet = wb[sheet_name]
        print(f"[Domain-based] Processing sheet: {sheet_name}")
        
        # 카테고리 값 -> 해당 행들의 리스트
        category_map = {}

        for row in sheet.iter_rows(values_only=True):
            if not row or all(cell is None for cell in row):
                continue

            cat_val = row[category_col] if category_col < len(row) else None
            cat_val = str(cat_val) if cat_val is not None else "UnknownCategory"

            row_str_list = [str(cell) if cell is not None else '' for cell in row]
            row_text = " | ".join(row_str_list).strip()

            if cat_val not in category_map:
                category_map[cat_val] = []
            category_map[cat_val].append(row_text)

        # category_map 에는 {카테고리: [행1, 행2, ...]} 형태
        # 각 카테고리별로 한 개의 chunk 로 만들기
        for cat_val, rows in category_map.items():
            chunk_text = f"[Sheet: {sheet_name} / Category: {cat_val}]\n" + "\n".join(rows)
            chunks.append(chunk_text)

    wb.close()
    return chunks


def detect_file_type(filename: str) -> str:
    """
    파일명(확장자)으로 Word/Excel/PDF/기타 텍스트를 대략적으로 구분.
    실제로는 MIME 타입, 매직 넘버 등을 확인하는 방법도 있음.
    """
    lower_name = filename.lower()
    if lower_name.endswith('.docx'):
        return 'word'
    elif lower_name.endswith('.xlsx') or lower_name.endswith('.xls'):
        return 'excel'
    elif lower_name.endswith('.pdf'):
        return 'pdf'
    else:
        return 'text'  # 그 외는 일반 텍스트로 처리


def read_and_chunk_file(
    file_bytes: bytes,
    filename: str,
    chunk_strategy: str,
    chunk_size: int,
    overlap_size: int,
    category_col: int = 0
) -> List[str]:
    """
    파일 확장자(또는 chunkStrategy)에 따라
    Word/Excel/PDF/일반 텍스트로 분기하여 텍스트 추출 + 청킹.
    
    - "heading": Word 문서에서 Heading 기반
    - "domain": Excel에서 특정 열(column) 기반으로 도메인별 청킹
    - "token": 단순 공백 분리(naive token) 기반 (dynamic_chunk_text)
    """
    file_type = detect_file_type(filename)
    print(f"Detected file type: {file_type}")

    # Word
    if file_type == 'word':
        if chunk_strategy == "heading":
            return chunk_word_heading(file_bytes, chunk_size, overlap_size)
        else:
            return chunk_word(file_bytes, chunk_size, overlap_size)

    # Excel
    elif file_type == 'excel':
        if chunk_strategy == "domain":
            return chunk_excel_domain_based(file_bytes, category_col=category_col)
        else:
            return chunk_excel(file_bytes, chunk_size, overlap_size)

    # PDF
    elif file_type == 'pdf':
        return chunk_pdf(file_bytes, chunk_size, overlap_size)

    # 일반 텍스트
    else:
        decoded_text = file_bytes.decode('utf-8', errors='ignore')
        return dynamic_chunk_text(decoded_text, strategy=chunk_strategy,
                                  chunk_size=chunk_size, overlap_size=overlap_size)


def generate_embeddings(text: List[str], model_id: str) -> List[np.ndarray]:
    """Amazon Bedrock의 invoke_model API를 사용하여 텍스트 임베딩 생성."""
    print(f"Generating embeddings for {len(text)} chunks using model {model_id}...")
    bedrock_runtime = boto3.client('bedrock-runtime', region_name='ap-northeast-2')
    embeddings = []

    for i, chunk in enumerate(text):
        try:
            print(f"[{i + 1}/{len(text)}] Processing chunk: {chunk[:50]}...")
            cleaned_chunk = chunk.replace('"', '\\"').strip()
            if not cleaned_chunk:
                print(f"Skipping empty or invalid chunk at index {i}.")
                continue

            payload = {
                "inputText": cleaned_chunk
            }

            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                contentType='application/json',
                accept='application/json',
                body=json.dumps(payload)
            )

            response_body = json.loads(response['body'].read())
            if 'embedding' not in response_body:
                raise ValueError("Response does not contain 'embedding' field.")

            embedding = np.array(response_body['embedding'], dtype=np.float32)
            embeddings.append(embedding)
            print(f"Chunk {i + 1} processed successfully.")

        except Exception as e:
            print(f"Error processing chunk {i + 1}: {e}")
            continue

    print("Embeddings generation completed.")
    return embeddings


def create_faiss_index(embeddings: List[np.ndarray]) -> faiss.IndexFlatL2:
    """FAISS 인덱스를 생성."""
    print("Creating FAISS index...")
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.vstack(embeddings))
    print("FAISS index created.")
    return index


def retrieve_relevant_chunks(query: str, faiss_index: faiss.IndexFlatL2, texts: List[str], model_id: str, top_k: int = 3) -> List[str]:
    """질문과 관련된 텍스트 청크 검색."""
    print(f"Retrieving relevant chunks for query: {query}")
    query_embedding = generate_embeddings([query], model_id=model_id)[0]
    distances, indices = faiss_index.search(np.array([query_embedding], dtype=np.float32), top_k)
    print(f"Retrieved indices: {indices[0]}, distances: {distances[0]}")
    return [texts[i] for i in indices[0] if i < len(texts)]


def lambda_handler(event, context):
    print("Lambda function invoked...")
    try:
        body = json.loads(event.get('body', '{}'))
        print("Request body:", body)

        # fileContent (Base64) 여부 확인
        file_content_base64 = body.get('fileContent', None)
        filename = body.get('fileName', 'default.md')  # 실제 파일 이름
        new_message = body.get('message', '스노우플레이크 문제 한개만 객관식으로 만들어줘')
        history = body.get('chatHistoryMessage', [])
        selectedModel = body.get('selectedModel', 'claude3.5')

        # ------------------ 동적 청킹 관련 파라미터 ------------------
        chunk_strategy = body.get('chunkStrategy', 'fixed')
        max_chunk_size = body.get('maxChunkSize', 500)
        if 'overlapSize' not in body or body['overlapSize'] is None:
            overlap_size = int(max_chunk_size * 0.1)
            print(f"overlapSize not provided. Using 10% of maxChunkSize => {overlap_size}")
        else:
            overlap_size = body['overlapSize']

        category_col = body.get('categoryCol', 0)
        # ----------------------------------------------------------

        print(f"chunkStrategy={chunk_strategy}, maxChunkSize={max_chunk_size}, overlapSize={overlap_size}, categoryCol={category_col}")

        embedding_model_id = 'amazon.titan-embed-text-v2:0'
        kbId = 'ZXCWNTBUPU'

        # Bedrock Foundation Model ARN 결정
        if selectedModel == 'claude3.5':
            modelArn = 'arn:aws:bedrock:ap-northeast-2::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0'
        elif selectedModel == 'claude3.0':
            modelArn = 'arn:aws:bedrock:ap-northeast-2::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0'
        else:
            raise ValueError(f"Unsupported model selected: {selectedModel}")

        print(f"Selected model ARN: {modelArn}")
        relevant_chunks = []  # 기본값: 빈 리스트

        # fileContent가 있으면 Base64 디코딩 + 청킹/임베딩/검색까지 진행
        if file_content_base64:
            try:
                file_content = base64.b64decode(file_content_base64)
            except Exception as e:
                raise ValueError(f"Error decoding base64 file content: {str(e)}")

            # 청킹
            print("Processing uploaded file...")
            text_chunks = read_and_chunk_file(
                file_bytes=file_content,
                filename=filename,
                chunk_strategy=chunk_strategy,
                chunk_size=max_chunk_size,
                overlap_size=overlap_size,
                category_col=category_col
            )

            print(f"Total chunks created: {len(text_chunks)}")

            # 파일에서 얻은 청크가 있다면 임베딩 + 검색
            if text_chunks:
                embeddings = generate_embeddings(text_chunks, embedding_model_id)
                faiss_index = create_faiss_index(embeddings)
                relevant_chunks = retrieve_relevant_chunks(new_message, faiss_index, text_chunks, embedding_model_id)
        else:
            print("No fileContent provided. Skipping file chunking and retrieval steps.")

        # 최종 모델에 넣을 프롬프트 템플릿 구성
        prompt_template = {
            "system_prompt": "AWS Bedrock 모델로서 사용자의 질문에 정중하고 정확하게 답변해야 합니다.",
            "context": {"goal": "사용자의 질문에 대해 가장 정확하고 관련 있는 정보를 제공하는 것입니다."},
            "conversation": history if isinstance(history, list) else []
        }

        # 사용자의 메시지와, 검색된 청크(있는 경우) 추가
        prompt_template["conversation"].append({"role": "user", "content": new_message})
        if relevant_chunks:
            prompt_template["context"]["retrieved_chunks"] = relevant_chunks

        # Bedrock Agent Runtime을 통해 RAG 호출
        response = bedrock_agent_runtime.retrieve_and_generate(
            input={'text': json.dumps(prompt_template)},
            retrieveAndGenerateConfiguration={
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': kbId,
                    'modelArn': modelArn
                },
                'type': 'KNOWLEDGE_BASE'
            }
        )

        output_text = response.get('output', {}).get('text', 'No output generated')
        print("Generated response:", output_text)

        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                "responseCode": "200",
                "responseStatus": "OK",
                "resultData": {
                    "message": output_text
                }
            })
        }

    except Exception as e:
        print("Error occurred:", str(e))
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'error': str(e)
            })
        }
