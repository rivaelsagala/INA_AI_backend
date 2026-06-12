# INA AI — Medical RAG Backend

Backend API untuk asisten kesehatan berbasis AI yang menggunakan Retrieval-Augmented Generation (RAG) untuk menjawab pertanyaan medis berdasarkan dataset penyakit umum di Indonesia.

## Fitur Utama

| #   | Fitur                                                   | Status |
| --- | ------------------------------------------------------- | ------ |
| 1   | RAG Pipeline (Hybrid Retrieval + Re-ranking)            | ✅     |
| 2   | Medical Content Guardrails                              | ✅     |
| 3   | PII Detection & Redaction                               | ✅     |
| 4   | Inference Endpoint (edge case handling)                 | ✅     |
| 5   | Evaluation Framework (RAGAS + LLM-as-Judge)             | ✅     |
| 6   | Multiple Retrieval Strategies (BM25 + Dense + Reranker) | ✅     |
| 7   | Cost Per Query Tracking                                 | ✅     |
| 8   | Inline Source Citations ([1], [2], ...)                 | ✅     |
| 9   | LLM-as-Judge Medical Evaluation (4 Kriteria)            | ✅     |
| 10  | Confidence Calibration + Overconfidence Detection       | ✅     |

## Tech Stack

- **Framework**: Flask + Gunicorn
- **Database**: Supabase (PostgreSQL + pgvector)
- **Embedding**: OpenAI `text-embedding-3-large` via Maia Router
- **LLM**: Multi-model (Llama 3.1 8B, Qwen 2.5 7B, DeepSeek R1, GPT-4o-mini)
- **Reranker**: Cohere Rerank v3.5
- **Evaluation**: RAGAS + Custom Medical Judge
- **Deployment**: Render

## Arsitektur

```
User Query → Handler (edge cases) → Use Case (orchestration)
                                        │
                                        ├── Guardrails Check
                                        ├── PII Redaction
                                        └── RAG Pipeline
                                              │
                                              ├── BM25 Retrieval (lexical)
                                              ├── Dense Retrieval (vector)
                                              ├── Hybrid Ensemble (50:50)
                                              ├── Cohere Reranker → top-5
                                              ├── Confidence Calibration
                                              ├── Citation Numbering
                                              ├── LLM Generation
                                              └── Cost Calculation
```

## Struktur Direktori

```
BE/
├── app/
│   ├── handler/          # HTTP request handlers + edge case validation
│   ├── usecases/         # Business logic orchestration
│   ├── services/         # Core services
│   │   ├── rag_service.py           # RAG pipeline utama (hybrid search + reranking + LLM)
│   │   ├── reranker_service.py      # Cohere Rerank API integration
│   │   ├── guardrails_service.py    # Regex-based harmful intent detection
│   │   ├── pii_service.py           # PII detection & redaction
│   │   ├── cost_service.py          # Cost tracking per query
│   │   ├── confidence_service.py    # Confidence calibration dari reranker scores
│   │   ├── ragas_service.py         # RAGAS evaluation integration
│   │   ├── chat_history_service.py  # Chat session management
│   │   ├── embedding_service.py     # Document embedding
│   │   └── preprocessing_service.py # CSV/PDF preprocessing
│   ├── routes.py         # API route definitions
│   └── __init__.py       # Flask app factory
├── eval/
│   ├── run_eval.py              # Evaluation runner (RAGAS + Medical Judge)
│   ├── medical_judge.py         # LLM-as-Judge (4 kriteria medis + bias validation)
│   ├── calibration_analysis.py  # Confidence calibration analysis + ECE
│   └── results/                 # Evaluation output (JSON + TXT)
├── data/
│   ├── csv/              # Dataset penyakit (sumber utama)
│   ├── pdf/              # Dokumen medis tambahan
│   └── processed/        # Data yang sudah diproses
├── requirements.txt
├── run.py                # Entrypoint aplikasi
└── runtime.txt           # Python version (Render)
```

## Setup & Instalasi

### 1. Clone & Install

```bash
git clone https://github.com/rivaelsagala/INA_AI_backend.git
cd BE
pip install -r requirements.txt
```

### 2. Environment Variables

Buat file `.env` di root:

```env
# Supabase
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJhbGci...
SUPABASE_TABLE_NAME=documents

# LLM API
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.maiarouter.ai/v1
HF_BASE_URL=https://router.huggingface.co/...
HF_TOKEN=hf_...

# Reranker
COHERE_API_KEY=...
```

### 3. Jalankan Server

```bash
python run.py
```

Server berjalan di `http://localhost:5000`.

## API Endpoints

| Method   | Endpoint                          | Deskripsi                   |
| -------- | --------------------------------- | --------------------------- |
| `POST`   | `/api/chat`                       | Kirim pesan ke Medical AI   |
| `POST`   | `/api/chat-sessions`              | Buat sesi baru              |
| `GET`    | `/api/chat-sessions?user_id=1`    | List semua sesi             |
| `GET`    | `/api/chat-history/<session_id>`  | Riwayat chat per sesi       |
| `PUT`    | `/api/chat-sessions/<session_id>` | Update nama sesi            |
| `DELETE` | `/api/chat-sessions/<session_id>` | Hapus sesi                  |
| `POST`   | `/api/ingest-medical-data`        | Ingest data CSV ke Supabase |

### Contoh Request — `/api/chat`

```json
POST /api/chat
{
  "message": "Apa gejala demam berdarah?",
  "session_id": 1,
  "user_id": 1,
  "model_id": 4
}
```

### Contoh Response

```json
{
  "status": "success",
  "answer": "Gejala demam berdarah dengue meliputi demam tinggi mendadak [1]...",
  "citations": [
    {
      "id": "[1]",
      "source": "dataset_penyakit.csv",
      "excerpt": "demam tinggi mendadak hingga 40 derajat...",
      "relevance_score": 0.9234
    }
  ],
  "cost": {
    "total_cost_usd": "0.0022198300",
    "estimated_cost_1000_queries": 2.2198
  },
  "confidence": {
    "confidence_score": 0.78,
    "confidence_label": "high"
  },
  "guardrail_triggered": false,
  "pii_detected": false
}
```

## Evaluation

### Menjalankan Evaluasi

```bash
# RAGAS + Medical Judge (full evaluation)
python eval/run_eval.py

# Calibration Analysis
python eval/calibration_analysis.py
```

### Metrik Evaluasi

**RAGAS Metrics:**

- Faithfulness — Anti-halusinasi (apakah jawaban berdasarkan konteks)
- Context Precision — Relevansi dokumen yang di-retrieve
- Answer Relevancy — Apakah jawaban menjawab pertanyaan

**Medical Judge (LLM-as-Judge):**

- Clinical Accuracy (1-5) — Keakuratan informasi klinis
- Safety Compliance (1-5) — Kepatuhan aturan keselamatan
- Source Grounding (1-5) — Apakah jawaban berbasis konteks
- Completeness (1-5) — Kelengkapan jawaban

**Confidence Calibration:**

- ECE (Expected Calibration Error)
- Overconfidence detection per bin

Hasil evaluasi disimpan di `eval/results/` dalam format JSON dan TXT.

## Safety & Privacy

- **Guardrails**: Mendeteksi intent berbahaya (self-harm, overdose, drug abuse, poisoning) dan memblokir sebelum masuk pipeline RAG
- **PII Redaction**: Mendeteksi dan menyamarkan NIK, nomor HP, email, nama orang dari input user
- **Disclaimer**: Setiap jawaban menyertakan saran untuk konsultasi dengan tenaga medis profesional
- **Confidence Disclaimer**: Ditampilkan otomatis jika confidence score rendah/sedang

## Model yang Didukung

| model_id | Model                       | Deskripsi              |
| -------- | --------------------------- | ---------------------- |
| 1        | Llama 3.1 8B Instruct       | Meta, open-source      |
| 2        | Qwen 2.5 7B Instruct        | Alibaba, open-source   |
| 3        | DeepSeek R1 Distill Qwen 7B | DeepSeek, open-source  |
| 4        | GPT-4o-mini                 | OpenAI via Maia Router |
