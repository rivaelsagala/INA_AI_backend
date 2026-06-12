import os
import re
import csv
import json
import psycopg2
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

def clean_medical_text(text: str) -> str:
    """
    Fungsi Preprocessing NLP untuk teks kesehatan.
    """
    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s\.,;:\-\(\)\/\%]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_text(text: str) -> list:
    """
    Tokenization: Memecah kalimat menjadi kata-kata tunggal.
    """
    return text.split()

def extract_and_preprocess_csv(file_path: str):
    """
    Ekstraksi data dari file CSV dan menerapkan text preprocessing.
    Mengembalikan tuple (raw_documents, preprocessed_documents).
    """
    raw_documents = []
    preprocessed_documents = []
    
    try:
        with open(file_path, mode='r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            
            for row_idx, row in enumerate(reader):
                keys = list(row.keys())
                penyakit_key = next((k for k in keys if k and "penyakit" in k.lower()), keys[0] if keys else "Unknown")
                nama_penyakit = str(row.get(penyakit_key, "Unknown")).strip()
                
                base_metadata = {
                    "source": os.path.basename(file_path), 
                    "title": f"Data Penyakit: {nama_penyakit}",
                    "row_index": row_idx + 1,
                    "nama_penyakit": nama_penyakit,
                    "type": "medical_knowledge"
                }
                
                # --- Tahap 1: Ekstraksi Mentah (Raw) ---
                raw_parts = []
                for key, value in row.items():
                    if value and str(value).strip():
                        raw_parts.append(f"{key}: {str(value).strip()}")
                raw_content = "\n".join(raw_parts)
                
                if raw_content.strip():
                    raw_documents.append(Document(
                        page_content=raw_content,
                        metadata=base_metadata
                    ))
                
                # --- Tahap 2: Preprocessing (Cleaned) ---
                cleaned_parts = []
                for key, value in row.items():
                    if value and str(value).strip():
                        cleaned_value = clean_medical_text(str(value))
                        cleaned_parts.append(f"{key.lower()}: {cleaned_value}")
                cleaned_content = "\n".join(cleaned_parts)
                
                if cleaned_content.strip():
                    preprocessed_documents.append(Document(
                        page_content=cleaned_content, 
                        metadata=base_metadata
                    ))

    except Exception as e:
        logger.error(f"Error membaca file CSV: {e}")
        
    return raw_documents, preprocessed_documents

def extract_and_preprocess_pdf(file_path: str):
    """
    Ekstraksi data dari file PDF menggunakan PyPDFLoader.
    Mengembalikan tuple (raw_documents, preprocessed_documents).
    """
    raw_documents = []
    preprocessed_documents = []
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        for i, page in enumerate(pages):
            base_metadata = {
                "source": os.path.basename(file_path),
                "title": f"Document: {os.path.basename(file_path)}",
                "page_number": i + 1,
                "type": "medical_knowledge"
            }
            
            # --- Tahap 1: Ekstraksi Mentah (Raw) ---
            raw_content = page.page_content
            if raw_content.strip():
                raw_documents.append(Document(
                    page_content=raw_content,
                    metadata=base_metadata
                ))
            
            # --- Tahap 2: Preprocessing (Cleaned) ---
            cleaned_content = clean_medical_text(page.page_content)
            if cleaned_content.strip():
                preprocessed_documents.append(Document(
                    page_content=cleaned_content,
                    metadata=base_metadata
                ))
    except Exception as e:
        logger.error(f"Error membaca file PDF: {e}")
    return raw_documents, preprocessed_documents

def create_medical_documents(file_path: str):
    """
    Fungsi sentral penentu ekstensi file untuk membuat dan memecah dokumen.
    Mengembalikan list chunks yang siap di-embed.
    """
    logger.info(f"Memulai preprocessing untuk file: {file_path}")
    
    if file_path.lower().endswith('.csv'):
        raw_docs, preprocessed_docs = extract_and_preprocess_csv(file_path)
    elif file_path.lower().endswith('.pdf'):
        raw_docs, preprocessed_docs = extract_and_preprocess_pdf(file_path)
    else:
        logger.error("Format file tidak didukung. Harap gunakan CSV atau PDF.")
        return []
        
    if not preprocessed_docs:
        return []
        
    # Chunking dilakukan pada hasil preprocessing
    chunks = chunk_documents(preprocessed_docs)
    
    # Simpan semua tahap ke folder per dokumen
    save_results_to_folder(file_path, raw_docs, preprocessed_docs, chunks)
    
    return chunks

def chunk_documents(documents: list):
    """
    5. Chunking
    Memecah dokumen teks yang besar ke dalam chunk yang lebih kecil dan padat.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n", ". ", ", ", " ", ""]
    )
    
    return text_splitter.split_documents(documents)

def save_results_to_folder(file_path: str, raw_docs: list, preprocessed_docs: list, chunks: list):
    """
    Menyimpan hasil setiap tahap pipeline ke dalam subfolder per dokumen:
      data/processed/<nama_dokumen>/
        ├── 1_extracted.txt        (Hasil ekstraksi mentah)
        ├── 2_preprocessed.txt     (Hasil setelah preprocessing/cleaning)
        └── 3_chunks.json          (Hasil chunking + tokenisasi)
    """
    base_filename = os.path.basename(file_path).rsplit('.', 1)[0]
    doc_dir = os.path.join(os.getcwd(), 'data', 'processed', base_filename)
    os.makedirs(doc_dir, exist_ok=True)
    
    # === 1. Simpan Hasil Ekstraksi Mentah (Raw) ===
    raw_path = os.path.join(doc_dir, f"{base_filename}_1_extracted.txt")
    with open(raw_path, 'w', encoding='utf-8') as f:
        for doc in raw_docs:
            idx = doc.metadata.get('row_index', doc.metadata.get('page_number', 'N/A'))
            info = doc.metadata.get('nama_penyakit', doc.metadata.get('title', 'N/A'))
            f.write(f"--- DOKUMEN {idx} : {info} ---\n")
            f.write(doc.page_content)
            f.write("\n\n")
    logger.info(f"[1/3] Hasil ekstraksi mentah disimpan: {raw_path}")
    
    # === 2. Simpan Hasil Preprocessing (Cleaned) ===
    preprocessed_path = os.path.join(doc_dir, f"{base_filename}_2_preprocessed.txt")
    with open(preprocessed_path, 'w', encoding='utf-8') as f:
        for doc in preprocessed_docs:
            idx = doc.metadata.get('row_index', doc.metadata.get('page_number', 'N/A'))
            info = doc.metadata.get('nama_penyakit', doc.metadata.get('title', 'N/A'))
            f.write(f"--- DOKUMEN {idx} : {info} ---\n")
            f.write(doc.page_content)
            f.write("\n\n")
    logger.info(f"[2/3] Hasil preprocessing disimpan: {preprocessed_path}")
    
    # === 3. Simpan Hasil Chunking & Tokenisasi (JSON) ===
    json_path = os.path.join(doc_dir, f"{base_filename}_3_chunks.json")
    chunks_data = []
    for i, chunk in enumerate(chunks):
        tokens = tokenize_text(chunk.page_content)
        chunks_data.append({
            "chunk_index": i + 1,
            "metadata": chunk.metadata,
            "character_count": len(chunk.page_content),
            "token_count": len(tokens),
            "tokens": tokens,
            "content": chunk.page_content
        })
        
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=4)
    logger.info(f"[3/3] Hasil chunking disimpan: {json_path}")
    
    logger.info(f"Pipeline selesai! Semua hasil disimpan di: {doc_dir}")

def save_chunks_to_postgres(chunks: list) -> bool:
    """
    Menyimpan data hasil chunking ke dalam tabel chunks_perdes di PostgreSQL.
    """
    conn = None
    cursor = None
    
    try:
        logger.info("Menghubungkan ke PostgreSQL database...")
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", ""),
            port=int(os.getenv("DB_PORT", "")),
            database=os.getenv("DB_NAME", ""),
            user=os.getenv("DB_USER", ""),
            password=os.getenv("DB_PASSWORD", "")
        )
        
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'chunks_perdes'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            logger.error("Tabel 'chunks_perdes' tidak ditemukan di database!")
            return False
        
        insert_query = """
        INSERT INTO chunks_perdes (file_name, content)
        VALUES (%s, %s)
        """
        
        inserted_count = 0
        for i, chunk in enumerate(chunks):
            try:
                file_name = chunk.metadata.get("source", "Unknown_Source")
                content = chunk.page_content
                
                cursor.execute(insert_query, (file_name, content))
                inserted_count += 1
                
            except Exception as chunk_error:
                logger.error(f"Error menyimpan chunk {i+1}: {chunk_error}")
                conn.rollback()
                return False

        conn.commit()
        logger.info(f"COMMIT berhasil! Total {inserted_count} chunks tersimpan di PostgreSQL")
        return True
        
    except psycopg2.Error as e:
        logger.error(f"Error Database PostgreSQL: {e}")
        return False
    except Exception as e:
        logger.error(f"Error tidak terduga saat menyimpan ke PostgreSQL: {e}")
        return False
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()

def extract_and_chunk_pdf(file_path: str):
    """
    Fungsi Utama Legacy.
    """
    return create_medical_documents(file_path)