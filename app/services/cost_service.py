"""
Cost Tracking Service untuk Medical RAG Pipeline.

Menghitung estimasi biaya (USD) per query berdasarkan:
1. Embedding cost (OpenAI text-embedding-3-large)
2. LLM Generation cost (per model)
3. Reranker cost (Cohere Rerank v3.5)

Referensi pricing: harga per 1M tokens (per Juni 2025).
"""

from typing import Dict, Any, Optional
from loguru import logger



MODEL_PRICING = {
    "openai/gpt-4o-mini": {
        "input_per_1m": 0.150,    # $0.15 per 1M input tokens
        "output_per_1m": 0.600,   # $0.60 per 1M output tokens
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "input_per_1m": 0.050,
        "output_per_1m": 0.050,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "input_per_1m": 0.050,
        "output_per_1m": 0.050,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "input_per_1m": 0.050,
        "output_per_1m": 0.050,
    },
}

EMBEDDING_PRICING = {
    "openai/text-embedding-3-large": {
        "per_1m": 0.130,  # $0.13 per 1M tokens
    }
}

RERANKER_PRICING = {
    "rerank-v3.5": {
        "per_1k_searches": 2.00,  # $2.00 per 1000 search units (Cohere)
    }
}


CHARS_PER_TOKEN = 4


def _fmt(value: float) -> str:
    """Format float kecil sebagai string agar tidak muncul scientific notation (1.43e-06) di JSON."""
    return f"{value:.10f}"


def estimate_tokens(text: str) -> int:
    """Estimasi jumlah token dari teks (heuristik: 1 token ≈ 4 karakter)."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def calculate_query_cost(
    query: str,
    context_texts: list,
    answer: str,
    model_name: str = "openai/gpt-4o-mini",
    num_reranked_docs: int = 15,
    system_prompt_length: int = 1500,
) -> Dict[str, Any]:
    """
    Menghitung estimasi biaya satu query RAG pipeline.

    Args:
        query: Pertanyaan user
        context_texts: List teks konteks yang dikirim ke LLM
        answer: Jawaban LLM
        model_name: Nama model LLM yang digunakan
        num_reranked_docs: Jumlah dokumen yang di-rerank
        system_prompt_length: Estimasi panjang karakter system prompt

    Returns:
        Dictionary dengan breakdown biaya per komponen
    """
    try:
        query_tokens = estimate_tokens(query)
        embedding_price = EMBEDDING_PRICING.get(
            "openai/text-embedding-3-large", {"per_1m": 0.130}
        )
        embedding_cost = (query_tokens / 1_000_000) * embedding_price["per_1m"]

        # LLM GENERATION COST
        # Input = system prompt + context + user query
        context_combined = "\n\n".join(context_texts) if context_texts else ""
        input_text_length = system_prompt_length + len(context_combined) + len(query)
        input_tokens = estimate_tokens("x" * input_text_length)  # approximate
        output_tokens = estimate_tokens(answer)

        model_price = MODEL_PRICING.get(
            model_name,
            {"input_per_1m": 0.050, "output_per_1m": 0.050} 
        )
        llm_input_cost = (input_tokens / 1_000_000) * model_price["input_per_1m"]
        llm_output_cost = (output_tokens / 1_000_000) * model_price["output_per_1m"]
        llm_total_cost = llm_input_cost + llm_output_cost

        # RERANKER COST (Cohere)
        # Cohere Rerank: 1 search unit = 1 query × N documents
        reranker_price = RERANKER_PRICING.get(
            "rerank-v3.5", {"per_1k_searches": 2.00}
        )
        # 1 search = reranking 1 query terhadap N docs
        reranker_cost = (1 / 1_000) * reranker_price["per_1k_searches"]


        # TOTAL
        total_cost = embedding_cost + llm_total_cost + reranker_cost

        cost_breakdown = {
            "embedding": {
                "tokens": query_tokens,
                "cost_usd": _fmt(embedding_cost),
            },
            "llm_generation": {
                "model": model_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost_usd": _fmt(llm_input_cost),
                "output_cost_usd": _fmt(llm_output_cost),
                "total_cost_usd": _fmt(llm_total_cost),
            },
            "reranker": {
                "model": "rerank-v3.5",
                "documents_reranked": num_reranked_docs,
                "cost_usd": _fmt(reranker_cost),
            },
            "total_cost_usd": _fmt(total_cost),
            "estimated_cost_1000_queries": round(total_cost * 1000, 4),
            "estimated_cost_10000_queries": round(total_cost * 10000, 4),
        }

        logger.debug(f"Cost breakdown: total=${total_cost:.6f} per query")
        return cost_breakdown

    except Exception as e:
        logger.error(f"Error calculating cost: {str(e)}")
        return {
            "error": str(e),
            "total_cost_usd": 0,
            "estimated_cost_1000_queries": 0,
        }


def get_pricing_info() -> Dict[str, Any]:
    """Mengembalikan tabel pricing yang digunakan untuk kalkulasi."""
    return {
        "llm_models": MODEL_PRICING,
        "embedding": EMBEDDING_PRICING,
        "reranker": RERANKER_PRICING,
        "estimation_method": "Token estimation: 1 token ≈ 4 characters (heuristic)",
        "note": "Harga berdasarkan pricing publik per Juni 2025. Aktual bisa berbeda tergantung tier dan volume."
    }
