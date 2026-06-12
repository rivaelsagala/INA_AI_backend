import os
import requests
from typing import Optional, Dict, Any, List
from loguru import logger
from supabase import create_client, Client
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from app.services.reranker_service import rerank_documents
from app.services.cost_service import calculate_query_cost
from app.services.confidence_service import calculate_confidence, format_confidence_disclaimer

# Load environment variables dari file .env
load_dotenv()

supabase_url = os.getenv("SUPABASE_URL", "")
supabase_key = os.getenv("SUPABASE_KEY", "")

supabase: Client = create_client(supabase_url, supabase_key)

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY", ""),
    base_url=os.getenv("OPENAI_BASE_URL", "")
)

AVAILABLE_MODELS = {
    1: {"name": "meta-llama/Llama-3.1-8B-Instruct", "type": "original"},
    2: {"name": "Qwen/Qwen2.5-7B-Instruct", "type": "original"},
    3: {"name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "type": "original"},
    4: {"name": "openai/gpt-4o-mini", "type": "openai"},
}

class HuggingFaceService:
    """Service untuk berinteraksi dengan HuggingFace Router API dan Fine-tuned Model"""
    
    def __init__(self):
        # HuggingFace Router API (Model belum fine-tuned)
        # Menggunakan HF_BASE_URL sesuai dengan .env Anda
        self.api_url = os.getenv("HF_BASE_URL", "")
        self.token = os.getenv("HF_TOKEN", "")
        
        
        # Common settings
        self.temperature = 0.0
        self.max_tokens = 2000
        
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    def query(self, messages: List[Dict[str, str]], model_id: int = 1, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Kirim query ke HuggingFace Router API, Fine-tuned Model API, atau Maia Router
        
        Args:
            messages: List of message dictionaries with role and content
            model_id: ID model yang akan digunakan (1-5)
                1: meta-llama/Llama-3.1-8B-Instruct (HuggingFace)
                2: Qwen/Qwen2.5-7B-Instruct (HuggingFace)
                3: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (HuggingFace)
                4: model_merged_legal (Fine-tuned)
                5: openai/gpt-4o-mini (Maia Router)
            **kwargs: Additional parameters like temperature, max_tokens
        """
        try:
            # Dapatkan info model, default ke model id 1 (Llama 3.1) jika id tidak ditemukan
            model_info = AVAILABLE_MODELS.get(model_id, AVAILABLE_MODELS[1])
            model_type = model_info.get("type", "original")
            
            if model_type == "openai":
                base_url = os.getenv("OPENAI_BASE_URL", "https://api.maiarouter.ai/v1").rstrip('/')
                api_url = f"{base_url}/chat/completions"
                api_key = os.getenv("OPENAI_API_KEY", "")
                
                logger.debug(f"Using MAIA ROUTER model: {model_info['name']}")
                
                payload = {
                    "model": model_info["name"],
                    "messages": messages,
                    "temperature": kwargs.get("temperature", self.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                }
                
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                logger.debug(f"Sending request to Maia Router: {api_url}")
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=300
                )
                response.raise_for_status()
                result = response.json()
                
                logger.info("Maia Router API response successful")
                return result
                
            else:
                api_url = self.api_url
                model_name = model_info["name"]
                logger.debug(f"Using ORIGINAL model: {model_name}")
                
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", self.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                }
                
                logger.debug(f"Sending request to HuggingFace: {api_url}")
                
                response = requests.post(
                    api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=300
                )
                
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"Model API response successful")
                return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API Error: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"Error detail: {error_detail}")
                except:
                    logger.error(f"Error response text: {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in query(): {str(e)}")
            return None
    
    def get_completion(self, messages: List[Dict[str, str]], model_id: int = 1, **kwargs) -> Optional[str]:
        """Dapatkan completion text dari messages"""
        response = self.query(messages, model_id=model_id, **kwargs)
        
        if response and "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
        
        logger.warning("No valid response content from LLM API")
        return None
    
    def chat_with_context(
        self,
        user_question: str,
        context: str,
        system_prompt: Optional[str] = None,
        model_id: int = 1,
        **kwargs
    ) -> Optional[str]:
        """
        Chat dengan context (untuk RAG)
        """
        if not system_prompt:
            system_prompt = f"""Anda adalah Asisten Kesehatan AI yang membantu menjawab pertanyaan tentang penyakit, gejala, dan pengobatan berdasarkan pengetahuan medis yang tersedia.

PEDOMAN UTAMA:
1. Jawab HANYA berdasarkan informasi dalam KONTEKS DOKUMEN yang diberikan
2. Gunakan bahasa Indonesia yang mudah dipahami
3. Berikan informasi yang akurat dan terstruktur
4. Selalu sarankan konsultasi dengan dokter/tenaga medis profesional
5. WAJIB sertakan nomor referensi [1], [2], dst. saat mengutip informasi dari konteks dokumen

FORMAT JAWABAN:
- Untuk pertanyaan tentang penyakit: jelaskan deskripsi, gejala, dan pengobatan yang tersedia
- Untuk pertanyaan tentang obat: sebutkan jenis obat dan dosis UMUM dari dokumen
- Untuk pertanyaan tentang gejala: identifikasi kemungkinan penyakit berdasarkan gejala tersebut
- WAJIB: Sertakan citation [1], [2], [3] dst. sesuai nomor sumber dokumen yang Anda gunakan
  Contoh: "Paracetamol diberikan 500 mg tiap 4-6 jam [1]."

ATURAN KESELAMATAN MEDIS (WAJIB):
⚠️ JANGAN PERNAH:
- Memberikan dosis obat untuk tujuan overdose atau membahayakan diri
- Memberikan instruksi cara bunuh diri atau menyakiti diri sendiri
- Memberikan cara meracuni orang lain
- Memberikan cara membuat atau menyalahgunakan narkotika
- Memberikan kombinasi obat yang berbahaya

✅ SELALU:
- Tambahkan disclaimer: "Konsultasikan dengan dokter sebelum menggunakan obat"
- Jika pertanyaan berbahaya, tolak dengan empati dan berikan nomor darurat:
  * Layanan Darurat Medis: 119
  * Hotline Kesehatan Jiwa: 500-454 atau (021) 500-454
  * IGD Rumah Sakit terdekat
- Untuk dosis obat, berikan HANYA informasi dari dokumen + peringatan konsultasi dokter

JIKA INFORMASI TIDAK ADA DALAM DOKUMEN:
Jawab: "Maaf, informasi tentang [topik] tidak tersedia dalam database kesehatan saya. Silakan konsultasikan dengan dokter atau tenaga medis profesional untuk informasi lebih lanjut."

KONTEKS DOKUMEN:
{context}

PERTANYAAN USER:"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]
        
        model_info = AVAILABLE_MODELS.get(model_id, AVAILABLE_MODELS[1])
        logger.info(f"Calling Model API ({model_info['name']}) with question: {user_question[:50]}...")
        return self.get_completion(messages, model_id=model_id, **kwargs)

# Singleton instance
hf_service = HuggingFaceService()

def get_answer_from_rag(query: str, model_id: int = 1) -> dict:
    """
    Mengeksekusi full pipeline RAG Medis dengan Multiple Retrieval Strategies:
    Hybrid Search (BM25 + Dense Retrieval) yang dilanjutkan dengan Re-ranking.
    
    Response mencakup:
    - answer: Jawaban LLM dengan inline citation [1], [2], dst.
    - sources: List sumber dokumen yang digunakan
    - citations: Mapping nomor citation ke sumber dokumen
    - cost: Breakdown biaya per komponen API
    - confidence: Skor kepercayaan berdasarkan reranker scores
    """
    supabase_table = os.getenv("SUPABASE_TABLE_NAME", "documents")
    

    # DENSE RETRIEVAL (Vector Search)

    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name=supabase_table,
        query_name="match_documents"
    )
    dense_retriever = vector_store.as_retriever(search_kwargs={"k": 15})
    
    # BM25 RETRIEVAL (Lexical Search)
    logger.info("Mempersiapkan BM25 Retriever dari Supabase...")
    bm25_retriever = None
    try:
        response = supabase.table(supabase_table).select("content, metadata").execute()
        all_docs = []
        if response.data:
            for row in response.data:
                page_content = row.get('content', '')
                metadata = row.get('metadata', {})
                all_docs.append(Document(page_content=page_content, metadata=metadata))
        
        if all_docs:
            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_retriever.k = 15
    except Exception as e:
        logger.error(f"Gagal memuat dokumen untuk BM25: {str(e)}")

    # HYBRID SEARCH (Ensemble)
    logger.info("Tahap 1 & 2: Mengambil dokumen kandidat menggunakan Hybrid Search (BM25 + Dense)...")
    if bm25_retriever:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[0.5, 0.5] # BM25 lebih tinggi jika exact match penting
        )
        initial_docs = ensemble_retriever.invoke(query)
    else:
        logger.warning("Fallback ke Dense Retrieval saja karena inisialisasi BM25 gagal.")
        initial_docs = dense_retriever.invoke(query)
        
    logger.info(f"Berhasil mengambil {len(initial_docs)} dokumen unik dari Hybrid Search.")
    
    # RE-RANKING (Cohere Rerank API)
    final_k = 5
    logger.info("Tahap 3: Menerapkan metode Re-ranking menggunakan Cohere Rerank API...")
    reranked_docs, relevance_scores = rerank_documents(query=query, documents=initial_docs, top_k=final_k)
    
    # CONFIDENCE CALIBRATION
    confidence_result = calculate_confidence(
        reranker_scores=relevance_scores,
        num_initial_docs=len(initial_docs),
        num_final_docs=len(reranked_docs)
    )
    confidence_disclaimer = format_confidence_disclaimer(confidence_result.get("confidence_label", "low"))
    
    # SOURCE ATTRIBUTION (Numbered Context)
    #  nomor pada setiap konteks agar LLM bisa cite [1], [2], dst.
    context_texts = []
    sources = []
    citations = []
    for idx, doc in enumerate(reranked_docs):
        citation_num = idx + 1
        context_texts.append(f"[{citation_num}] {doc.page_content}")
        
        source_name = doc.metadata.get('source', doc.metadata.get('nama_penyakit', 'Unknown'))
        
        sources.append({"content": doc.page_content, "metadata": doc.metadata})
        citations.append({
            "id": f"[{citation_num}]",
            "source": source_name,
            "excerpt": doc.page_content[:150] + ("..." if len(doc.page_content) > 150 else ""),
            "relevance_score": round(relevance_scores[idx], 4) if idx < len(relevance_scores) else None
        })
        
    context_joined = "\n\n---\n\n".join(context_texts)

    # LLM GENERATION
    model_info = AVAILABLE_MODELS.get(model_id, AVAILABLE_MODELS[1])
    logger.info(f"Using Model {model_info['name']} for Generation")
    
    answer = hf_service.chat_with_context(
        user_question=query,
        context=context_joined,
        model_id=model_id
    )
    
    final_answer = answer if answer else "Maaf, terjadi kesalahan saat mencoba menghasilkan jawaban dari model bahasa."


    # COST CALCULATION
    cost_breakdown = calculate_query_cost(
        query=query,
        context_texts=[doc.page_content for doc in reranked_docs],
        answer=final_answer,
        model_name=model_info["name"],
        num_reranked_docs=len(initial_docs)
    )

    return {
        "answer": final_answer,
        "sources": sources,
        "citations": citations,
        "model_used": model_info["name"],
        "cost": cost_breakdown,
        "confidence": confidence_result,
        "confidence_disclaimer": confidence_disclaimer
    }