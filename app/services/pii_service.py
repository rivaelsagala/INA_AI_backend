import re
from typing import Tuple, Dict, List


# =============================================================
# REGEX PATTERNS UNTUK DETEKSI PII
# =============================================================

PII_PATTERNS = {
    # 1. NIK Indonesia: tepat 16 digit angka, berdiri sendiri (bukan bagian dari angka lain)
    "ID_NUM": re.compile(r"(?<!\d)[1-9]\d{15}(?!\d)"),

    # 2. Nomor HP Indonesia: harus diawali +62/08 dan total 10-13 digit
    #    Lebih strict: wajib prefix 08 atau +62, bukan angka biasa
    "PHONE_NUMBER": re.compile(
        r"(?:\+62|0062)"          # prefix internasional
        r"[\s\-]?"
        r"\d{8,12}\b"
        r"|"
        r"\b08\d{8,11}\b"        # prefix lokal 08xx (10-13 digit total)
    ),

    # 3. Email
    "EMAIL_ADDRESS": re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    ),

    # 4. Nama orang (heuristik bahasa Indonesia)
    #    Menangkap pola: "nama saya Budi Santoso", "pasien Ahmad", dll.
    "PERSON": re.compile(
        r"(?:nama\s+saya|saya\s+(?:bernama|adalah)|pasien(?:\s+bernama)?|atas\s+nama|an[./]\s*)"
        r"\s+"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",  # 1-4 kata kapital
        re.IGNORECASE
    ),
}

# Pattern angka/teks medis yang BUKAN PII (untuk menghindari false positive)
MEDICAL_EXCLUDE_PATTERNS = re.compile(
    r"(?:"
    r"\d+\s*(?:mg|ml|mcg|gram|g|kg|tablet|kapsul|sendok|tetes|cc|iu|unit)"  # dosis obat
    r"|\d+[\-\.]\d+"                     # range/angka desimal (misal: 500-454, 0.5)
    r"|119|500[\-\s]454|021[\-\s]\d+"    # nomor darurat/hotline
    r"|ICD[\-\s]?\d+"                    # kode ICD
    r")",
    re.IGNORECASE
)


def _find_pii_spans(text: str) -> List[dict]:
    """
    Scan teks dan kembalikan list span PII yang ditemukan.
    Setiap item: {"entity_type": str, "start": int, "end": int, "text": str}
    """
    spans = []

    for entity_type, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            matched_text = match.group()
            
            # Skip jika match-nya adalah konteks medis (dosis, hotline, dll)
            if MEDICAL_EXCLUDE_PATTERNS.search(matched_text):
                continue
                
            # Cek konteks sekitar (±20 karakter) untuk false positive medis
            ctx_start = max(0, match.start() - 20)
            ctx_end = min(len(text), match.end() + 20)
            context = text[ctx_start:ctx_end]
            if MEDICAL_EXCLUDE_PATTERNS.search(context):
                continue
            
            if entity_type == "PERSON":
                # Untuk PERSON, kita ambil group(1) yaitu nama-nya saja
                if match.lastindex and match.group(1):
                    spans.append({
                        "entity_type": entity_type,
                        "start": match.start(1),
                        "end": match.end(1),
                        "text": match.group(1)
                    })
            else:
                spans.append({
                    "entity_type": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group()
                })

    # Urutkan dari posisi terakhir ke awal agar replacement tidak geser index
    spans.sort(key=lambda x: x["start"], reverse=True)

    # Hapus overlap (jika ada span yang tumpang tindih, ambil yang lebih panjang)
    filtered: List[dict] = []
    for span in spans:
        overlapping = False
        for existing in filtered:
            if span["start"] < existing["end"] and span["end"] > existing["start"]:
                overlapping = True
                break
        if not overlapping:
            filtered.append(span)

    return filtered


def redact_pii(text: str) -> Tuple[str, Dict[str, int]]:
    """
    Mendeteksi dan me-redact PII menggunakan regex patterns.

    Returns:
        Tuple of (teks_yang_sudah_di_redact, summary_dict)
        summary_dict berisi jumlah per tipe entitas, contoh: {"PHONE_NUMBER": 1, "EMAIL_ADDRESS": 1}
    """
    if not text:
        return text, {}

    spans = _find_pii_spans(text)

    if not spans:
        return text, {}

    # Buat summary
    summary: Dict[str, int] = {}
    for span in spans:
        entity_type = span["entity_type"]
        summary[entity_type] = summary.get(entity_type, 0) + 1

    # Lakukan redaction (ganti dengan placeholder)
    redacted_text = text
    for span in spans:  # sudah diurutkan descending
        placeholder = f"<{span['entity_type']}>"
        redacted_text = (
            redacted_text[:span["start"]]
            + placeholder
            + redacted_text[span["end"]:]
        )

    return redacted_text, summary


def has_pii(text: str) -> bool:
    """Cek boolean sederhana apakah ada PII atau tidak."""
    if not text:
        return False
    spans = _find_pii_spans(text)
    return len(spans) > 0


def safe_log(text: str) -> str:
    """Helper untuk me-redact text sebelum masuk logging standar."""
    redacted_text, _ = redact_pii(text)
    # Potong jika terlalu panjang untuk log
    if len(redacted_text) > 200:
        return redacted_text[:200] + "..."
    return redacted_text