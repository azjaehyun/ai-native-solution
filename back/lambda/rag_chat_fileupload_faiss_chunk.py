import boto3
import json
import faiss
import numpy as np
import base64
from typing import List
from io import BytesIO
import sys


print(sys.path)


# (1) Word 파일 처리용 라이브러리
try:
    from docx import Document
    print("docx 라이브러리 import 성공!")
except ImportError as e:
    print(f"docx ImportError 발생: {e}")

# (2) PDF 파일 처리용 라이브러리
try:
    import PyPDF2
    print("PyPDF2 라이브러리 import 성공!")
except ImportError:
    pass

# (3) Excel 파일 처리용 라이브러리
try:
    from openpyxl import load_workbook
    print("openpyxl 라이브러리 import 성공!")
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
                start = 0
                while start < len(para):
                    end = start + chunk_size
                    chunk = para[start:end]
                    chunks.append(chunk)
                    start += max(chunk_size - overlap_size, 1)
        return chunks

    elif strategy == "sentence":
        import re
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
        # 간단히 공백 기준으로 나눈 뒤 chunk_size 단위로 쪼개는 예
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
        # fixed
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
        if 'Heading' in style_name:
            if current_heading or current_text_buffer:
                chunks.extend(flush_chunk(current_heading, current_text_buffer))
            current_heading = text
            current_text_buffer = []
        else:
            if not current_heading:
                current_heading = "No Heading"
            if text:
                current_text_buffer.append(text)

    if current_heading or current_text_buffer:
        chunks.extend(flush_chunk(current_heading, current_text_buffer))

    return chunks


def chunk_pdf(
    file_bytes: bytes,
    chunk_size: int = 500,
    overlap_size: int = 50
) -> List[str]:
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
    너무 길면 슬라이딩 윈도우(chunk_size/overlapSize)로 분할.

    디버깅을 위해:
    1) read_only=False, data_only=False 로 변경
    2) row_text 출력
    3) 필요시 헤더 스킵 로직 추가
    """
    chunks = []
    try:
        # 수정 부분: read_only=False, data_only=False
        wb = load_workbook(filename=BytesIO(file_bytes), read_only=False, data_only=False)
    except Exception as e:
        print(f"Error loading Excel: {e}")
        return []

    for sheet_name in wb.sheetnames:
        sheet = wb[sheet_name]
        print(f"Processing sheet: {sheet_name}")

        for row_idx, row in enumerate(sheet.iter_rows(values_only=True), start=1):

            # (선택) 헤더 스킵 로직 예시:
            # if row_idx == 1:
            #     print("Skipping header row.")
            #     continue

            row_str_list = [str(cell) if cell is not None else '' for cell in row]
            row_text = " | ".join(row_str_list).strip()

            print(f"Row {row_idx} raw text => '{row_text}'")
            if not row_text:
                print(f"Row {row_idx} is empty. Skipping.")
                continue

            if len(row_text) <= chunk_size:
                chunk_str = f"[Sheet: {sheet_name}] {row_text}"
                print(f"Row {row_idx} => One chunk => {chunk_str}")
                chunks.append(chunk_str)
            else:
                start = 0
                while start < len(row_text):
                    end = start + chunk_size
                    chunk = row_text[start:end]
                    chunk_str = f"[Sheet: {sheet_name}] {chunk}"
                    print(f"Row {row_idx} => Splitted chunk => {chunk_str}")
                    chunks.append(chunk_str)
                    start += max(chunk_size - overlap_size, 1)

    wb.close()
    return chunks


def chunk_excel_domain_based(
    file_bytes: bytes,
    category_col: int = 0
) -> List[str]:
    chunks = []
    try:
        # domain-based에서는 여전히 필요에 따라 data_only=False 사용 가능
        wb = load_workbook(filename=BytesIO(file_bytes), read_only=False, data_only=False)
    except Exception as e:
        print(f"Error loading Excel for domain chunking: {e}")
        return []

    for sheet_name in wb.sheetnames:
        sheet = wb[sheet_name]
        print(f"[Domain-based] Processing sheet: {sheet_name}")
        
        category_map = {}

        for row_idx, row in enumerate(sheet.iter_rows(values_only=True), start=1):
            # (선택) 헤더 스킵
            # if row_idx == 1:
            #     continue

            if not row or all(cell is None for cell in row):
                continue

            cat_val = row[category_col] if category_col < len(row) else None
            cat_val = str(cat_val) if cat_val is not None else "UnknownCategory"

            row_str_list = [str(cell) if cell is not None else '' for cell in row]
            row_text = " | ".join(row_str_list).strip()

            if cat_val not in category_map:
                category_map[cat_val] = []
            category_map[cat_val].append(row_text)

        for cat_val, rows in category_map.items():
            chunk_text = f"[Sheet: {sheet_name} / Category: {cat_val}]\n" + "\n".join(rows)
            chunks.append(chunk_text)

    wb.close()
    return chunks


def detect_file_type(filename: str) -> str:
    lower_name = filename.lower()
    if lower_name.endswith('.docx'):
        return 'word'
    elif lower_name.endswith('.xlsx') or lower_name.endswith('.xls'):
        return 'excel'
    elif lower_name.endswith('.pdf'):
        return 'pdf'
    else:
        return 'text'


def chunk_word_full_text(
    file_bytes: bytes,
    chunk_size: int = 500,
    overlap_size: int = 50
) -> List[str]:
    try:
        doc = Document(BytesIO(file_bytes))
        print("[INFO] Word document successfully loaded.")
    except Exception as e:
        print(f"[ERROR] Error loading Word docx: {e}")
        return []

    all_text = []

    # Collect text from paragraphs
    print("[INFO] Extracting paragraphs from the document...")
    for i, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if text:
            all_text.append(text)
            print(f"[DEBUG] Paragraph {i + 1}: {text}")

    # Include table content if any
    print("[INFO] Extracting tables from the document...")
    for table_idx, table in enumerate(doc.tables):
        print(f"[DEBUG] Table {table_idx + 1} found.")
        for row_idx, row in enumerate(table.rows):
            row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_text:
                table_row_text = " | ".join(row_text)
                all_text.append(table_row_text)
                print(f"[DEBUG] Table {table_idx + 1}, Row {row_idx + 1}: {table_row_text}")

    # Collect text from headers and footers if any
    print("[INFO] Extracting headers and footers from the document...")
    if doc.sections:
        for section_idx, section in enumerate(doc.sections):
            print(f"[DEBUG] Section {section_idx + 1} found.")
            header = section.header
            footer = section.footer
            if header:
                for para_idx, para in enumerate(header.paragraphs):
                    text = para.text.strip()
                    if text:
                        header_text = f"[Header] {text}"
                        all_text.append(header_text)
                        print(f"[DEBUG] Section {section_idx + 1} Header Paragraph {para_idx + 1}: {header_text}")
            if footer:
                for para_idx, para in enumerate(footer.paragraphs):
                    text = para.text.strip()
                    if text:
                        footer_text = f"[Footer] {text}"
                        all_text.append(footer_text)
                        print(f"[DEBUG] Section {section_idx + 1} Footer Paragraph {para_idx + 1}: {footer_text}")

    # Combine all extracted text
    print("[INFO] Combining all extracted text...")
    combined_text = "\n".join(all_text)
    print(f"[DEBUG] Combined Text Length: {len(combined_text)} characters")

    # Split into chunks
    print("[INFO] Splitting text into chunks...")
    chunks = []
    start = 0
    while start < len(combined_text):
        end = start + chunk_size
        chunk = combined_text[start:end]
        chunks.append(chunk)
        print(f"[DEBUG] Chunk {len(chunks)}: {chunk[:50]}...")  # Print first 50 chars of each chunk
        start += max(chunk_size - overlap_size, 1)

    print(f"[INFO] Total Chunks Created: {len(chunks)}")
    return chunks

def read_and_chunk_file(
    file_bytes: bytes,
    filename: str,
    chunk_strategy: str,
    chunk_size: int,
    overlap_size: int,
    category_col: int = 0
) -> List[str]:
    file_type = detect_file_type(filename)
    print(f"Detected file type: {file_type}")

    if file_type == 'word':
        if chunk_strategy == "heading":
            return chunk_word_heading(file_bytes, chunk_size, overlap_size)
        elif chunk_strategy == "full_text":  # 추가된 옵션
            return chunk_word_full_text(file_bytes, chunk_size, overlap_size)
        else:
            return chunk_word(file_bytes, chunk_size, overlap_size)

    elif file_type == 'excel':
        if chunk_strategy == "domain":
            return chunk_excel_domain_based(file_bytes, category_col=category_col)
        else:
            return chunk_excel(file_bytes, chunk_size, overlap_size)

    elif file_type == 'pdf':
        return chunk_pdf(file_bytes, chunk_size, overlap_size)

    else:
        decoded_text = file_bytes.decode('utf-8', errors='ignore')
        return dynamic_chunk_text(decoded_text, strategy=chunk_strategy,
                                  chunk_size=chunk_size, overlap_size=overlap_size)


def generate_embeddings(text: List[str], model_id: str) -> List[np.ndarray]:
    print(f"Generating embeddings for {len(text)} chunks using model {model_id}...")
    bedrock_runtime = boto3.client('bedrock-runtime', region_name='ap-northeast-2')
    embeddings = []

    for i, chunk in enumerate(text):
        try:
            #print(f"[{i + 1}/{len(text)}] Processing chunk: {chunk[:50]}...")
            print(f"[{i + 1}/{len(text)}] Processing chunk: {chunk} ")
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
    print("Creating FAISS index...")
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.vstack(embeddings))
    print("FAISS index created.")
    return index


def retrieve_relevant_chunks(query: str, faiss_index: faiss.IndexFlatL2, texts: List[str], model_id: str, top_k: int = 3) -> List[str]:
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

        file_content_base64 = body.get('fileContent', None)
        filename = body.get('fileName', 'default.md')
        new_message = body.get('message', '스노우플레이크 문제 한개만 객관식으로 만들어줘')
        history = body.get('chatHistoryMessage', [])
        selectedModel = body.get('selectedModel', 'claude3.5')

        chunk_strategy = body.get('chunkStrategy', 'full_text')  #  fixed
        max_chunk_size = body.get('maxChunkSize', 500)
        if 'overlapSize' not in body or body['overlapSize'] is None:
            overlap_size = int(max_chunk_size * 0.1)
            print(f"overlapSize not provided. Using 10% of maxChunkSize => {overlap_size}")
        else:
            overlap_size = body['overlapSize']

        category_col = body.get('categoryCol', 0)

        print(f"chunkStrategy={chunk_strategy}, maxChunkSize={max_chunk_size}, overlapSize={overlap_size}, categoryCol={category_col}")

        embedding_model_id = 'amazon.titan-embed-text-v2:0'
        kbId = 'ZXCWNTBUPU'

        if selectedModel == 'claude3.5':
            modelArn = 'arn:aws:bedrock:ap-northeast-2::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0'
        elif selectedModel == 'claude3.0':
            modelArn = 'arn:aws:bedrock:ap-northeast-2::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0'
        else:
            raise ValueError(f"Unsupported model selected: {selectedModel}")

        print(f"Selected model ARN: {modelArn}")
        relevant_chunks = []

        if file_content_base64:
            try:
                file_content = base64.b64decode(file_content_base64)
            except Exception as e:
                raise ValueError(f"Error decoding base64 file content: {str(e)}")

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

            if text_chunks:
                embeddings = generate_embeddings(text_chunks, embedding_model_id)
                faiss_index = create_faiss_index(embeddings)
                relevant_chunks = retrieve_relevant_chunks(new_message, faiss_index, text_chunks, embedding_model_id)
        else:
            print("No fileContent provided. Skipping file chunking and retrieval steps.")

        prompt_template = {
            "system_prompt": "AWS Bedrock 모델로서 사용자의 질문에 정중하고 정확하게 답변해야 합니다.",
            "context": {"goal": "사용자의 질문에 대해 가장 정확하고 관련 있는 정보를 제공하는 것입니다."},
            "conversation": history if isinstance(history, list) else []
        }

        prompt_template["conversation"].append({"role": "user", "content": new_message})
        if relevant_chunks:
            prompt_template["context"]["retrieved_chunks"] = relevant_chunks

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
