import os
import sys
import json
from datetime import datetime
from loguru import logger

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.ragas_service import ragas_service

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def save_results(results: dict, questions: list, answers: list, contexts_list: list, ground_truths: list):
    """Simpan hasil evaluasi ke folder eval/results dalam format JSON dan TXT."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(RESULTS_DIR, f"eval_{timestamp}.json")
    txt_path = os.path.join(RESULTS_DIR, f"eval_{timestamp}.txt")

    full_payload = {
        "timestamp": datetime.now().isoformat(),
        "test_cases": [],
        "results": results,
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
    lines.append("   LAPORAN EVALUASI MEDICAL RAG (RAGAS)")
    lines.append(f"   Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    lines.append("")

    if "error" in results:
        lines.append(f"ERROR: {results['error']}")
    else:
        avg = results.get("average_scores", {})
        lines.append(">> SKOR RATA-RATA")
        lines.append(f"   Faithfulness (Akurasi Faktual) : {avg.get('faithfulness', 0):.4f}")
        lines.append(f"   Context Precision              : {avg.get('context_precision', 0):.4f}")
        lines.append(f"   Answer Relevancy               : {avg.get('answer_relevancy', 0):.4f}")
        lines.append(f"   SKOR KESELURUHAN               : {avg.get('average_score', 0):.4f}")
        lines.append("")

        individual = results.get("individual_scores", [])
        if individual:
            lines.append("-" * 60)
            lines.append(">> DETAIL PER PERTANYAAN")
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
    logger.info("Memulai Evaluasi Framework RAGAS untuk Medical AI...")

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

    results = ragas_service.evaluate_batch(
        questions=questions,
        answers=answers,
        contexts_list=contexts_list,
        ground_truths=ground_truths
    )

    print("\n" + "=" * 60)
    print("      HASIL EVALUASI MEDICAL RAG (RAGAS)")
    print("=" * 60)

    if "error" in results:
        print(f"ERROR: {results['error']}")
    else:
        avg_scores = results.get("average_scores", {})
        print(f"Rata-rata Faithfulness (Faktual)    : {avg_scores.get('faithfulness', 0):.4f}")
        print(f"Rata-rata Context Precision         : {avg_scores.get('context_precision', 0):.4f}")
        print(f"Rata-rata Answer Relevancy          : {avg_scores.get('answer_relevancy', 0):.4f}")
        print(f"SKOR RATA-RATA KESELURUHAN          : {avg_scores.get('average_score', 0):.4f}")

        individual = results.get("individual_scores", [])
        if individual:
            print("\n" + "-" * 60)
            print("  DETAIL PER PERTANYAAN:")
            print("-" * 60)
            for idx, score in enumerate(individual):
                print(f"\n  [{idx + 1}] {questions[idx]}")
                print(f"      Faithfulness      : {score.get('faithfulness', 0):.4f}")
                print(f"      Context Precision : {score.get('context_precision', 0):.4f}")
                print(f"      Answer Relevancy  : {score.get('answer_relevancy', 0):.4f}")

    print("=" * 60)

    json_path, txt_path = save_results(
        results=results,
        questions=questions,
        answers=answers,
        contexts_list=contexts_list,
        ground_truths=ground_truths
    )

    print(f"\nHasil disimpan di:")
    print(f"   JSON : {json_path}")
    print(f"   TXT  : {txt_path}")


if __name__ == "__main__":
    run_medical_eval()