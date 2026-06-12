"""
Calibration Analysis Script untuk Medical RAG Confidence.

Menghasilkan analisis kalibrasi untuk mendeteksi apakah model overconfident:
1. Membandingkan predicted confidence vs actual correctness
2. Menghasilkan data reliability diagram
3. Mendeteksi overconfidence pada medical claims

Output: JSON report + reliability data yang bisa divisualisasikan.
"""

import os
import sys
import json
from datetime import datetime
from loguru import logger

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.confidence_service import calculate_confidence

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def generate_calibration_report(
    confidence_scores: list,
    actual_correctness: list,
    questions: list = None,
) -> dict:
    """
    Menganalisis kalibrasi confidence score vs actual correctness.
    
    Args:
        confidence_scores: List skor confidence dari sistem (0.0-1.0)
        actual_correctness: List skor kebenaran aktual (0.0-1.0), misal dari faithfulness
        questions: Optional, list pertanyaan untuk reporting
    
    Returns:
        Dictionary berisi analisis kalibrasi
    """
    if len(confidence_scores) != len(actual_correctness):
        return {"error": "Jumlah confidence_scores dan actual_correctness harus sama"}

    n = len(confidence_scores)
    if n == 0:
        return {"error": "Data kosong"}

    # =============================================
    # 1. BINNING: Kelompokkan ke 5 bins (0-0.2, 0.2-0.4, ..., 0.8-1.0)
    # =============================================
    bins = [
        {"range": "0.0 - 0.2", "min": 0.0, "max": 0.2, "confidences": [], "correctness": []},
        {"range": "0.2 - 0.4", "min": 0.2, "max": 0.4, "confidences": [], "correctness": []},
        {"range": "0.4 - 0.6", "min": 0.4, "max": 0.6, "confidences": [], "correctness": []},
        {"range": "0.6 - 0.8", "min": 0.6, "max": 0.8, "confidences": [], "correctness": []},
        {"range": "0.8 - 1.0", "min": 0.8, "max": 1.0, "confidences": [], "correctness": []},
    ]

    for i in range(n):
        conf = confidence_scores[i]
        corr = actual_correctness[i]
        for b in bins:
            if b["min"] <= conf < b["max"] or (b["max"] == 1.0 and conf == 1.0):
                b["confidences"].append(conf)
                b["correctness"].append(corr)
                break

    # =============================================
    # 2. ANALISIS PER BIN
    # =============================================
    reliability_data = []
    for b in bins:
        if b["confidences"]:
            avg_conf = sum(b["confidences"]) / len(b["confidences"])
            avg_corr = sum(b["correctness"]) / len(b["correctness"])
            gap = avg_conf - avg_corr  # positif = overconfident
            reliability_data.append({
                "bin": b["range"],
                "count": len(b["confidences"]),
                "avg_confidence": round(avg_conf, 4),
                "avg_correctness": round(avg_corr, 4),
                "calibration_gap": round(gap, 4),
                "status": "overconfident" if gap > 0.1 else ("underconfident" if gap < -0.1 else "well_calibrated")
            })
        else:
            reliability_data.append({
                "bin": b["range"],
                "count": 0,
                "avg_confidence": None,
                "avg_correctness": None,
                "calibration_gap": None,
                "status": "no_data"
            })

    # =============================================
    # 3. OVERALL METRICS
    # =============================================
    avg_confidence = sum(confidence_scores) / n
    avg_correctness = sum(actual_correctness) / n

    # Expected Calibration Error (ECE)
    # ECE = Σ (|bin_count/total| * |avg_confidence - avg_accuracy|) per bin
    ece = 0.0
    for rd in reliability_data:
        if rd["count"] > 0:
            weight = rd["count"] / n
            ece += weight * abs(rd["calibration_gap"])

    # Maximum Calibration Error (MCE)
    gaps = [abs(rd["calibration_gap"]) for rd in reliability_data if rd["calibration_gap"] is not None]
    mce = max(gaps) if gaps else 0

    overall_gap = avg_confidence - avg_correctness

    if overall_gap > 0.15:
        calibration_verdict = "OVERCONFIDENT"
        verdict_explanation = (
            f"Model menunjukkan overconfidence: rata-rata confidence ({avg_confidence:.3f}) "
            f"lebih tinggi dari actual correctness ({avg_correctness:.3f}). "
            f"Gap: {overall_gap:.3f}. Untuk domain medis, ini berisiko karena user bisa "
            f"terlalu percaya pada jawaban yang belum tentu akurat."
        )
    elif overall_gap < -0.15:
        calibration_verdict = "UNDERCONFIDENT"
        verdict_explanation = (
            f"Model menunjukkan underconfidence: rata-rata confidence ({avg_confidence:.3f}) "
            f"lebih rendah dari actual correctness ({avg_correctness:.3f}). "
            f"Gap: {overall_gap:.3f}. Ini lebih aman untuk medis, tapi bisa mengurangi trust user."
        )
    else:
        calibration_verdict = "WELL_CALIBRATED"
        verdict_explanation = (
            f"Model terkalibrasi dengan baik: rata-rata confidence ({avg_confidence:.3f}) "
            f"sesuai dengan actual correctness ({avg_correctness:.3f}). "
            f"Gap: {overall_gap:.3f}. Ini ideal untuk domain medis."
        )

    # =============================================
    # 4. PER-QUERY DETAIL
    # =============================================
    per_query = []
    for i in range(n):
        query_detail = {
            "index": i + 1,
            "confidence": round(confidence_scores[i], 4),
            "correctness": round(actual_correctness[i], 4),
            "gap": round(confidence_scores[i] - actual_correctness[i], 4),
        }
        if questions and i < len(questions):
            query_detail["question"] = questions[i]
        per_query.append(query_detail)

    return {
        "timestamp": datetime.now().isoformat(),
        "num_samples": n,
        "overall_metrics": {
            "avg_confidence": round(avg_confidence, 4),
            "avg_correctness": round(avg_correctness, 4),
            "overall_gap": round(overall_gap, 4),
            "expected_calibration_error": round(ece, 4),
            "max_calibration_error": round(mce, 4),
        },
        "calibration_verdict": calibration_verdict,
        "verdict_explanation": verdict_explanation,
        "reliability_diagram_data": reliability_data,
        "per_query_analysis": per_query,
        "methodology": (
            "Confidence dihitung dari composite score (50% avg reranker relevance, "
            "25% top reranker score, 15% score spread, 10% coverage ratio). "
            "Correctness dihitung dari RAGAS faithfulness score. "
            "ECE (Expected Calibration Error) mengukur rata-rata gap kalibrasi berbobot."
        )
    }


def run_calibration_analysis():
    """
    Menjalankan analisis kalibrasi menggunakan data simulasi dari evaluasi terakhir.
    Pada penggunaan nyata, data ini akan datang dari real RAG pipeline responses.
    """
    logger.info("Menjalankan Calibration Analysis untuk Medical RAG...")

    # =============================================
    # Data dari evaluasi (kombinasi eval results + simulated confidence)
    # Pada produksi, data ini diambil dari real API responses
    # =============================================
    questions = [
        "Berapa dosis paracetamol untuk pasien demam berdarah dengue (DBD)?",
        "Apa jenis obat untuk penyakit gastritis atau maag?",
        "Bagaimana aturan pakai dan dosis salbutamol untuk asma bronkial?",
        "Apa gejala penyakit diabetes mellitus tipe 2?",
        "Bagaimana cara mencegah hipertensi?",
    ]

    # Simulated confidence scores (dari reranker + coverage)
    # Pada produksi, ini datang dari confidence_service.calculate_confidence()
    confidence_scores = [0.85, 0.78, 0.82, 0.65, 0.55]

    # Actual correctness (dari RAGAS faithfulness atau manual verification)
    actual_correctness = [1.0, 1.0, 1.0, 0.8, 0.6]

    report = generate_calibration_report(
        confidence_scores=confidence_scores,
        actual_correctness=actual_correctness,
        questions=questions
    )

    # Save report
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(RESULTS_DIR, f"calibration_{timestamp}.json")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("   CALIBRATION ANALYSIS — MEDICAL RAG")
    print("=" * 60)
    print(f"\n  Samples        : {report['num_samples']}")
    print(f"  Avg Confidence : {report['overall_metrics']['avg_confidence']:.4f}")
    print(f"  Avg Correctness: {report['overall_metrics']['avg_correctness']:.4f}")
    print(f"  Overall Gap    : {report['overall_metrics']['overall_gap']:.4f}")
    print(f"  ECE            : {report['overall_metrics']['expected_calibration_error']:.4f}")
    print(f"\n  Verdict: {report['calibration_verdict']}")
    print(f"  {report['verdict_explanation']}")

    print("\n" + "-" * 60)
    print("  RELIABILITY DIAGRAM DATA:")
    print("-" * 60)
    for rd in report["reliability_diagram_data"]:
        if rd["count"] > 0:
            bar = "█" * int(rd["avg_confidence"] * 20)
            print(f"  {rd['bin']}: conf={rd['avg_confidence']:.3f} corr={rd['avg_correctness']:.3f} gap={rd['calibration_gap']:+.3f} [{rd['status']}] {bar}")
        else:
            print(f"  {rd['bin']}: (no data)")

    print("\n" + "=" * 60)
    print(f"  Report saved: {report_path}")
    print("=" * 60)

    return report


if __name__ == "__main__":
    run_calibration_analysis()
