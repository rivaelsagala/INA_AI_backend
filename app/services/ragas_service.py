import os
import warnings
from typing import Dict, Any, List
from loguru import logger
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision
)
from dotenv import load_dotenv

# Import library Langchain untuk custom endpoint
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

# Import wrapper Ragas untuk standarisasi format
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

# Supaya warning tidak mengganggu
warnings.filterwarnings("ignore", category=DeprecationWarning)

class RagasEvaluationService:
    """
    Service untuk mengevaluasi respons Medical RAG menggunakan RAGAS metrics.
    
    Metrik Utama untuk Medical RAG:
    1. Faithfulness (Akurasi Faktual): Mengukur apakah jawaban akhir LLM benar-benar hanya 
       berasal dari konteks yang ditarik (dokumen medis), bukan dari "halusinasi".
       Di domain medis, AI boleh bilang "Saya tidak tahu", tapi haram mengarang fakta/dosis.
       
    2. Answer Relevancy: Mengukur seberapa tepat jawaban yang diberikan dengan pertanyaan awal pasien/user.
       
    3. Context Precision (Retrieval Quality): Mengukur apakah retriever (Vector DB) berhasil 
       menarik dokumen klinis yang tepat dan menempatkannya di urutan teratas.
    """
    
    def __init__(self):
        # Konfigurasi API dari env
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.base_url = os.getenv("OPENAI_BASE_URL", "")
        
        # 1. Inisialisasi LLM Langchain
        # Turunkan temperature ke 0.0 agar juri absolut, faktual dan tidak berubah-ubah
        langchain_llm = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model="openai/gpt-3.5-turbo-16k",
            temperature=0.0
        )
        
        # 2. Inisialisasi Embeddings Langchain
        langchain_embeddings = OpenAIEmbeddings(
            api_key=self.api_key,
            base_url=self.base_url,
            model="openai/text-embedding-3-large"
        )
        
        # 3. WAJIB: Bungkus LLM dan Embeddings dengan Wrapper bawaan RAGAS agar format JSON tidak rusak
        self.custom_llm = LangchainLLMWrapper(langchain_llm)
        self.custom_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
        
        # Metrik yang akan digunakan khusus Medical
        self.metrics = [
            faithfulness,           # Apakah jawaban didukung mutlak oleh literatur medis?
            answer_relevancy,       # Apakah jawaban relevan dengan pertanyaan pasien?
            context_precision       # Apakah dokumen klinis teratas tepat sasaran? (butuh ground_truth)
        ]
        
        logger.info("RAGAS Evaluation Service initialized for Medical Domain")
    
    def evaluate_response(
        self,
        question: str,
        answer: str,
        retrieved_contexts: List[str],
        ground_truth: str = None
    ) -> Dict[str, Any]:
        """
        Evaluasi satu respons Medical RAG
        
        Args:
            question: Pertanyaan user/pasien
            answer: Jawaban dari asisten medis AI
            retrieved_contexts: List konteks klinis yang diambil dari vector database
            ground_truth: Jawaban pakar medis (valid expert answer). Sangat direkomendasikan 
                          untuk metrik context_precision.
        """
        try:
            if ground_truth is None:
                logger.warning(
                    "'ground_truth' (jawaban pakar medis) tidak diberikan. "
                    "Metrik context_precision mungkin tidak seakurat jika dibandingkan dengan referensi medis."
                )
                ground_truth = answer
            
            # Siapkan dataset untuk evaluasi
            data_sample = {
                "question": [question],
                "contexts": [retrieved_contexts],
                "answer": [answer],
                "ground_truth": [ground_truth]
            }
            
            eval_dataset = Dataset.from_dict(data_sample)
            
            logger.info(f"🔍 Evaluating medical response for question: {question[:50]}...")
            
            # Jalankan evaluasi
            evaluation_result = evaluate(
                dataset=eval_dataset,
                metrics=self.metrics,
                llm=self.custom_llm,
                embeddings=self.custom_embeddings
            )
            
            # Konversi hasil ke dictionary
            df_results = evaluation_result.to_pandas()
            result_dict = df_results.iloc[0].to_dict()
            
            # Format hasil sesuai metrik yang kita gunakan
            formatted_result = {
                "faithfulness": float(result_dict.get("faithfulness", 0)),
                "answer_relevancy": float(result_dict.get("answer_relevancy", 0)),
                "context_precision": float(result_dict.get("context_precision", 0)),
            }
            
            # Rata-rata dari seluruh metrik
            formatted_result["average_score"] = round(
                sum(formatted_result.values()) / len(formatted_result), 4
            )
            
            logger.info(f"Evaluation completed: {formatted_result}")
            return formatted_result
            
        except Exception as e:
            logger.error(f"Error during Medical RAGAS evaluation: {str(e)}")
            return {
                "error": str(e),
                "faithfulness": 0,
                "answer_relevancy": 0,
                "context_precision": 0,
                "average_score": 0
            }
            
    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluasi batch untuk banyak pertanyaan medis sekaligus (Skenario Demo)
        """
        try:
            if ground_truths is None:
                logger.warning("Batch evaluation berjalan tanpa 'ground_truth' (jawaban medis pakar).")
                ground_truths = answers

            data_sample = {
                "question": questions,
                "contexts": contexts_list,
                "answer": answers,
                "ground_truth": ground_truths
            }
            
            eval_dataset = Dataset.from_dict(data_sample)
            logger.info(f"🔍 Running RAGAS batch evaluation on {len(questions)} test cases...")
            
            evaluation_result = evaluate(
                dataset=eval_dataset,
                metrics=self.metrics,
                llm=self.custom_llm,
                embeddings=self.custom_embeddings
            )
            
            df_results = evaluation_result.to_pandas()
            
            # Hitung rata-rata
            avg_scores = {
                "faithfulness": float(df_results["faithfulness"].mean()) if "faithfulness" in df_results else 0,
                "answer_relevancy": float(df_results["answer_relevancy"].mean()) if "answer_relevancy" in df_results else 0,
                "context_precision": float(df_results["context_precision"].mean()) if "context_precision" in df_results else 0,
            }
            
            avg_scores["average_score"] = round(sum(avg_scores.values()) / len(avg_scores), 4)
            
            return {
                "average_scores": avg_scores,
                "individual_scores": df_results.to_dict(orient="records")
            }
            
        except Exception as e:
            logger.error(f"Error during Medical Batch RAGAS evaluation: {str(e)}")
            return {"error": str(e)}
    
    def format_contexts_from_sources(self, sources: List[Dict[str, Any]]) -> List[str]:
        """
        Format sources dari RAG menjadi list of strings untuk RAGAS
        """
        contexts = []
        for source in sources:
            if isinstance(source, dict) and "content" in source:
                contexts.append(source["content"])
            elif isinstance(source, str):
                contexts.append(source)
        
        return contexts

# Singleton instance
ragas_service = RagasEvaluationService()

