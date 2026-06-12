import os
import sys
import json
from datetime import datetime
from loguru import logger

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.ragas_service import ragas_service
from eval.medical_judge import evaluate_batch_with_judge, get_judge_prompt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def save_results(results: dict, judge_results: dict, questions: list, answers: list, contexts_list: list, ground_truths: list):
    """Simpan hasil evaluasi (RAGAS + Medical Judge) ke folder eval/results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(RESULTS_DIR, f"eval_{timestamp}.json")
    txt_path = os.path.join(RESULTS_DIR, f"eval_{timestamp}.txt")

    full_payload = {
        "timestamp": datetime.now().isoformat(),
        "test_cases": [],
        "ragas_results": results,
        "medical_judge_results": judge_results,
    }

    for i, q in enumerate(questions):
        full_payload["test_cases"].append({
            "question": q,
            "answer": answers[i],
            "contexts": contexts_list[i],
            "ground_truth": ground_truths[i],
        })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, ensure_ascii=False, indent=2)

    logger.info(f"Hasil JSON disimpan di: {json_path}")

    lines = []
    lines.append("=" * 60)
    lines.append("   LAPORAN EVALUASI MEDICAL RAG")
    lines.append(f"   Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    lines.append("")

    # ---- RAGAS Section ----
    lines.append("━" * 60)
    lines.append("  BAGIAN 1: RAGAS METRICS")
    lines.append("━" * 60)

    if "error" in results:
        lines.append(f"ERROR: {results['error']}")
    else:
        avg = results.get("average_scores", {})
        lines.append(f"   Faithfulness (Akurasi Faktual) : {avg.get('faithfulness', 0):.4f}")
        lines.append(f"   Context Precision              : {avg.get('context_precision', 0):.4f}")
        lines.append(f"   Answer Relevancy               : {avg.get('answer_relevancy', 0):.4f}")
        lines.append(f"   SKOR KESELURUHAN               : {avg.get('average_score', 0):.4f}")
        lines.append("")

        individual = results.get("individual_scores", [])
        if individual:
            lines.append("-" * 60)
            lines.append(">> DETAIL PER PERTANYAAN (RAGAS)")
            lines.append("-" * 60)
            for idx, score in enumerate(individual):
                lines.append(f"\n[Test Case {idx + 1}]")
                lines.append(f"  Pertanyaan   : {questions[idx]}")
                lines.append(f"  Jawaban LLM  : {answers[idx][:120]}...")
                lines.append(f"  Ground Truth : {ground_truths[idx][:120]}...")
                lines.append(f"  ---")
                lines.append(f"  Faithfulness       : {score.get('faithfulness', 0):.4f}")
                lines.append(f"  Context Precision  : {score.get('context_precision', 0):.4f}")
                lines.append(f"  Answer Relevancy   : {score.get('answer_relevancy', 0):.4f}")

    # ---- Medical Judge Section ----
    lines.append("")
    lines.append("━" * 60)
    lines.append("  BAGIAN 2: LLM-AS-JUDGE (MEDICAL CRITERIA)")
    lines.append("━" * 60)

    if judge_results and "error" not in judge_results:
        agg = judge_results.get("aggregate_scores", {})
        lines.append(f"   Clinical Accuracy  : {agg.get('clinical_accuracy', 0):.2f} / 5.0")
        lines.append(f"   Safety Compliance  : {agg.get('safety_compliance', 0):.2f} / 5.0")
        lines.append(f"   Source Grounding   : {agg.get('source_grounding', 0):.2f} / 5.0")
        lines.append(f"   Completeness       : {agg.get('completeness', 0):.2f} / 5.0")
        lines.append(f"   OVERALL            : {agg.get('overall', 0):.2f} / 5.0")
        lines.append("")
        lines.append(f"   Judge Model    : {judge_results.get('judge_model', 'N/A')}")
        lines.append(f"   Bias Check     : {judge_results.get('bias_check_enabled', False)}")

        # Bias validation per case
        ind_results = judge_results.get("individual_results", [])
        for idx, ir in enumerate(ind_results):
            bv = ir.get("bias_validation", {})
            if bv:
                lines.append(f"\n   [Case {idx+1}] Reliability: {bv.get('reliability', 'N/A')} "
                             f"(variance: {bv.get('average_variance', 0):.4f})")
    else:
        lines.append("   (Medical Judge evaluation tidak dijalankan atau gagal)")

    lines.append("")
    lines.append("=" * 60)
    lines.append("  Evaluasi selesai.")
    lines.append("=" * 60)

    report_text = "\n".join(lines)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"Laporan TXT disimpan di: {txt_path}")

    return json_path, txt_path


def run_medical_eval():
    logger.info("Memulai Evaluasi Framework untuk Medical AI...")

    questions = [
        "Berapa dosis paracetamol untuk pasien demam berdarah dengue (DBD)?",
        "Apa jenis obat untuk penyakit gastritis atau maag?",
        "Bagaimana aturan pakai dan dosis salbutamol untuk asma bronkial?"
    ]

    contexts_list = [
        [
            "nama penyakit: demam berdarah dengue (dbd)\ngejala: demam tinggi mendadak hingga 40 derajat celsius\njenis obat: antipiretik (paracetamol)\ndosis obat: paracetamol: 500 mg tiap 4-6 jam jika demam."
        ],
        [
            "nama penyakit: gastritis (maag)\ndeskripsi: peradangan pada lapisan lambung.\njenis obat: antasida, proton pump inhibitor (omeprazole), h2 blocker (ranitidin)"
        ],
        [
            "nama penyakit: asma bronkial\njenis obat: bronkodilator (salbutamol inhaler)\ndosis obat: salbutamol: 1-2 isapan (puff) saat serangan asma terjadi, maksimal 4 kali sehari."
        ]
    ]

    answers = [
        "Untuk pasien demam berdarah dengue (DBD), dosis paracetamol adalah 500 mg tiap 4-6 jam jika terjadi demam.",
        "Jenis obat untuk gastritis atau maag meliputi antasida, proton pump inhibitor seperti omeprazole, dan h2 blocker seperti ranitidin.",
        "Dosis salbutamol untuk asma bronkial adalah 1-2 isapan (puff) saat serangan asma terjadi, dengan batas maksimal penggunaan 4 kali sehari."
    ]

    ground_truths = [
        "Paracetamol: 500 mg tiap 4-6 jam jika demam.",
        "Antasida, proton pump inhibitor (omeprazole), h2 blocker (ranitidin).",
        "Salbutamol: 1-2 isapan (puff) saat serangan asma terjadi, maksimal 4 kali sehari."
    ]

    # =====================================================
    # BAGIAN 1: RAGAS EVALUATION
    # =====================================================
    print("\n" + "=" * 60)
    print("  [1/2] Menjalankan RAGAS Evaluation...")
    print("=" * 60)

    results = ragas_service.evaluate_batch(
        questions=questions,
        answers=answers,
        contexts_list=contexts_list,
        ground_truths=ground_truths
    )

    if "error" in results:
        print(f"RAGAS ERROR: {results['error']}")
    else:
        avg_scores = results.get("average_scores", {})
        print(f"  Faithfulness      : {avg_scores.get('faithfulness', 0):.4f}")
        print(f"  Context Precision : {avg_scores.get('context_precision', 0):.4f}")
        print(f"  Answer Relevancy  : {avg_scores.get('answer_relevancy', 0):.4f}")
        print(f"  OVERALL           : {avg_scores.get('average_score', 0):.4f}")

    # =====================================================
    # BAGIAN 2: LLM-AS-JUDGE MEDICAL EVALUATION
    # =====================================================
    print("\n" + "=" * 60)
    print("  [2/2] Menjalankan Medical Judge Evaluation (dengan Bias Check)...")
    print("=" * 60)

    # Flatten contexts untuk judge (join list jadi string)
    contexts_flat = ["\n".join(ctx) for ctx in contexts_list]

    judge_results = evaluate_batch_with_judge(
        questions=questions,
        answers=answers,
        contexts_list=contexts_flat,
        ground_truths=ground_truths,
        with_bias_check=True
    )

    if "error" not in judge_results:
        agg = judge_results.get("aggregate_scores", {})
        print(f"  Clinical Accuracy  : {agg.get('clinical_accuracy', 0):.2f} / 5.0")
        print(f"  Safety Compliance  : {agg.get('safety_compliance', 0):.2f} / 5.0")
        print(f"  Source Grounding   : {agg.get('source_grounding', 0):.2f} / 5.0")
        print(f"  Completeness       : {agg.get('completeness', 0):.2f} / 5.0")
        print(f"  OVERALL            : {agg.get('overall', 0):.2f} / 5.0")

        # Print bias validation summary
        for idx, ir in enumerate(judge_results.get("individual_results", [])):
            bv = ir.get("bias_validation", {})
            if bv:
                print(f"\n  [Case {idx+1}] Reliability: {bv.get('reliability', 'N/A')} "
                      f"(avg variance: {bv.get('average_variance', 0):.4f})")
    else:
        print(f"  Judge ERROR: {judge_results.get('error', 'Unknown')}")

    # =====================================================
    # SAVE ALL RESULTS
    # =====================================================
    print("\n" + "=" * 60)

    json_path, txt_path = save_results(
        results=results,
        judge_results=judge_results,
        questions=questions,
        answers=answers,
        contexts_list=contexts_list,
        ground_truths=ground_truths
    )

    print(f"\nHasil disimpan di:")
    print(f"   JSON : {json_path}")
    print(f"   TXT  : {txt_path}")

    # Print judge prompt untuk transparency
    print("\n" + "-" * 60)
    print("  JUDGE PROMPT (untuk transparency/demo):")
    print("-" * 60)
    print(get_judge_prompt()[:500] + "...\n")


if __name__ == "__main__":
    run_medical_eval()
