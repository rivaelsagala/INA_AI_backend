import os
from loguru import logger
from cohere import Client
from dotenv import load_dotenv

load_dotenv()

COHERE_TOKEN = os.getenv("COHERE_API_KEY", "")

try:
    logger.info("Memuat Cohere Client untuk Re-ranking...")
    cohere_client = Client(
        base_url="https://api.cohere.com/",
        token=COHERE_TOKEN
    )
    logger.info("Cohere Client berhasil dimuat!")
except Exception as e:
    logger.error(f"Gagal memuat Cohere Client: {e}")
    cohere_client = None

def rerank_documents(query: str, documents: list, top_k: int = 5) -> list:
    """
    Melakukan re-ranking pada dokumen menggunakan Cohere Rerank API (v2).
    Sangat cocok untuk domain Medis agar konteks logis antara kueri pasien 
    dan literatur klinis dapat dicocokkan dengan presisi tinggi.
    
    Args:
        query: Pertanyaan dari user
        documents: List objek Document (kandidat awal dari Hybrid Search)
        top_k: Jumlah final dokumen yang akan diambil setelah diurutkan ulang
    """
    if not cohere_client or not documents:
        logger.warning("Cohere Client tidak tersedia atau dokumen kosong. Mengembalikan dokumen asli.")
        return documents[:top_k]
    
    docs_texts = [doc.page_content for doc in documents]
    
    logger.info(f"Melakukan re-ranking dengan Cohere untuk {len(documents)} dokumen kandidat medis...")
    
    try:
        actual_top_k = min(top_k, len(documents))
        
        response = cohere_client.v2.rerank(
            model="rerank-v4.0-pro",
            query=query,
            documents=docs_texts,
            top_n=actual_top_k
        )
        
        reranked_docs = []
        logger.debug(f"--- Top {actual_top_k} Hasil Re-ranking Medis (Cohere) ---")
        
        for rank, result in enumerate(response.results):
            original_idx = result.index
            score = result.relevance_score
            
            original_doc = documents[original_idx]
            
            source = original_doc.metadata.get('source', 'Unknown')
            page = original_doc.metadata.get('page', '?')
            logger.debug(f"Rank {rank+1} | Score: {score:.4f} | Source: {source} (Hal. {page})")
            
            reranked_docs.append(original_doc)
            
        return reranked_docs
        
    except Exception as e:
        logger.error(f"Error saat memanggil Cohere Rerank API: {e}")
        return documents[:top_k]