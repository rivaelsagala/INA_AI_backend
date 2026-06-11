import re
from typing import Dict, Any

# Dictionary kategori dan regex polanya
HARMFUL_PATTERNS = {
    "self_harm": r"\b(bunuh diri|mati|mengakhiri hidup|overdosis|overdose|od|menyilet|menyakiti diri)\b",
    "lethal_dosage": r"\b(dosis fatal|biar mati|dosis mematikan|berapa banyak.*(mati|lewat|overdosis))\b",
    "drug_abuse": r"\b(meracik|membuat (sabu|ekstasi|narkoba)|mabuk obat|fly|ngefly)\b",
    "poisoning": r"\b(meracuni|racun untuk|membunuh)\b"
}

def check_medical_guardrails(text: str) -> Dict[str, Any]:
    """
    Input Filter untuk memblokir pertanyaan medis yang membahayakan.
    Bekerja sebagai proxy layer sebelum masuk ke pipeline RAG.
    """
    text_lower = text.lower()
    
    for category, pattern in HARMFUL_PATTERNS.items():
        if re.search(pattern, text_lower):
            return {
                "is_blocked": True,
                "layer": "input_filter",
                "category": category,
                "reason": "Mendeteksi intensi membahayakan diri, overdosis, atau penyalahgunaan obat.",
                "refusal_message": (
                    "Saya adalah asisten AI dan tidak memiliki kapasitas untuk memberikan saran yang membahayakan, "
                    "panduan overdosis, atau diagnosis medis absolut. Jika Anda sedang dalam kondisi krisis atau "
                    "membutuhkan bantuan segera, mohon hubungi:\n\n"
                    "Layanan Gawat Darurat Medis: 119\n"
                    "Hotline Kesehatan Jiwa Kementerian Kesehatan: 500-454\n"
                    "Atau segera kunjungi IGD Rumah Sakit terdekat."
                )
            }
            
    # Jika aman
    return {
        "is_blocked": False,
        "layer": "input_filter",
        "category": "safe",
        "reason": "passed",
        "refusal_message": None
    }