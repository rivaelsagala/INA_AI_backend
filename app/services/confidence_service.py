"""
Confidence Calibration Service untuk Medical RAG Pipeline.

Menghitung skor kepercayaan (confidence score) berdasarkan:
1. Reranker relevance scores (Cohere)
2. Coverage ratio (berapa banyak konteks relevan yang ditemukan)
3. Query-context semantic alignment

Output:
- confidence_score (0.0 - 1.0)
- confidence_label (low / medium / high)
- confidence_reasoning (penjelasan)
"""

from typing import Dict, Any, List, Optional
from loguru import logger



CONFIDENCE_THRESHOLDS = {
    "high": 0.70,
    "medium": 0.40,
    # di bawah 0.40 = low
}


def calculate_confidence(
    reranker_scores: List[float],
    num_initial_docs: int = 15,
    num_final_docs: int = 5,
) -> Dict[str, Any]:
    """
    Menghitung composite confidence score dari hasil retrieval pipeline.
    
    Komponen:
    1. avg_relevance: Rata-rata skor relevansi dari reranker (0-1)
    2. top_score: Skor tertinggi dari reranker
    3. score_spread: Selisih antara skor tertinggi dan terendah (indikator kejelasan)
    4. coverage_ratio: Rasio dokumen final vs kandidat awal
    
    Args:
        reranker_scores: List skor relevansi dari Cohere Reranker (sudah urut desc)
        num_initial_docs: Jumlah dokumen kandidat sebelum reranking
        num_final_docs: Jumlah dokumen final setelah reranking
    
    Returns:
        Dictionary dengan confidence_score, label, dan reasoning
    """
    if not reranker_scores:
        return {
            "confidence_score": 0.0,
            "confidence_label": "low",
            "confidence_reasoning": "Tidak ada skor reranker yang tersedia.",
            "components": {}
        }

    try:

        #Average Relevance Score

        avg_relevance = sum(reranker_scores) / len(reranker_scores)
        

        # Top Score (skor dokumen paling relevan)

        top_score = max(reranker_scores)
        
        # Spread tinggi = model bisa membedakan dokumen relevan vs tidak
        if len(reranker_scores) > 1:
            score_spread = max(reranker_scores) - min(reranker_scores)
        else:
            score_spread = 0.0
        
        # Jika retriever menemukan banyak dokumen relevan, confidence lebih tinggi
        coverage_ratio = min(num_final_docs / max(num_initial_docs, 1), 1.0)
        

        # Weighted combination:
        # - 50% avg_relevance (seberapa relevan konteks yang dipakai)
        # - 25% top_score (apakah ada setidaknya 1 dokumen sangat relevan)
        # - 15% score_spread (apakah model bisa bedakan relevan/tidak)
        # - 10% coverage_ratio (seberapa banyak referensi tersedia)
        composite_score = (
            0.50 * avg_relevance +
            0.25 * top_score +
            0.15 * score_spread +
            0.10 * coverage_ratio
        )
        
        # Clamp ke 0-1
        composite_score = max(0.0, min(1.0, composite_score))
        

        if composite_score >= CONFIDENCE_THRESHOLDS["high"]:
            label = "high"
            reasoning = (
                f"Confidence tinggi. Reranker menemukan dokumen medis dengan relevansi kuat "
                f"(avg={avg_relevance:.3f}, top={top_score:.3f}). "
                f"Jawaban ini didukung oleh referensi medis yang memadai."
            )
        elif composite_score >= CONFIDENCE_THRESHOLDS["medium"]:
            label = "medium"
            reasoning = (
                f"Confidence sedang. Reranker menemukan beberapa dokumen relevan "
                f"(avg={avg_relevance:.3f}, top={top_score:.3f}), "
                f"namun relevansi tidak sepenuhnya kuat. "
                f"Disarankan untuk memverifikasi dengan tenaga medis profesional."
            )
        else:
            label = "low"
            reasoning = (
                f"Confidence rendah. Dokumen yang ditemukan memiliki relevansi terbatas "
                f"(avg={avg_relevance:.3f}, top={top_score:.3f}). "
                f"Sangat disarankan untuk berkonsultasi langsung dengan dokter atau tenaga medis."
            )
        
        result = {
            "confidence_score": round(composite_score, 4),
            "confidence_label": label,
            "confidence_reasoning": reasoning,
            "components": {
                "avg_relevance": round(avg_relevance, 4),
                "top_score": round(top_score, 4),
                "score_spread": round(score_spread, 4),
                "coverage_ratio": round(coverage_ratio, 4),
            }
        }
        
        logger.debug(
            f"Confidence: {composite_score:.4f} ({label}) | "
            f"avg_rel={avg_relevance:.3f} top={top_score:.3f} "
            f"spread={score_spread:.3f} coverage={coverage_ratio:.3f}"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating confidence: {str(e)}")
        return {
            "confidence_score": 0.0,
            "confidence_label": "low",
            "confidence_reasoning": f"Error dalam kalkulasi confidence: {str(e)}",
            "components": {}
        }


def format_confidence_disclaimer(confidence_label: str) -> Optional[str]:
    """
    Menghasilkan disclaimer tambahan berdasarkan level confidence.
    Untuk ditampilkan ke user sebagai peringatan.
    """
    disclaimers = {
        "low": (
            "Perhatian: Sistem memiliki tingkat keyakinan RENDAH terhadap jawaban ini. "
            "Informasi medis yang tersedia dalam database terbatas untuk pertanyaan ini. "
            "Sangat disarankan untuk berkonsultasi langsung dengan dokter atau tenaga medis profesional."
        ),
        "medium": (
            "Catatan: Sistem memiliki tingkat keyakinan SEDANG terhadap jawaban ini. "
            "Beberapa referensi medis ditemukan, namun mungkin tidak sepenuhnya mencakup pertanyaan Anda. "
            "Disarankan untuk memverifikasi informasi ini dengan tenaga medis profesional."
        ),
        "high": None 
    }
    return disclaimers.get(confidence_label)
