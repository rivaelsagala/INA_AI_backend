from flask import request, jsonify
from loguru import logger
from app.usecases.chat_use_case import (
    chat_with_history,
    create_new_session,
    get_user_sessions,
    get_session_history,
    update_session_name,
    delete_session
)

def handle_chat():
    """
    Handle chat request dengan guardrails, PII protection, dan edge case handling 
    khusus untuk domain Kesehatan (Poin 4: Robustness & Edge Cases).
    
    Edge cases yang di-handle:
    1. Empty string / whitespace only -> HTTP 400
    2. Very long text (>2000 characters) -> HTTP 413 (Payload Too Large)
    3. Code-switching ID/EN -> Dicatat untuk analitik
    4. Missing required fields
    5. Invalid model_id
    """
    try:
        data = request.get_json()
        
        # Validasi request body
        if not data:
            return jsonify({"error": "Request body kosong"}), 400
        
        # Check required fields
        required_fields = ['message', 'session_id', 'user_id']
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            return jsonify({
                "error": f"Field wajib tidak ada: {', '.join(missing_fields)}",
                "required_fields": required_fields
            }), 400
        
        message = data['message']
        session_id = data['session_id']
        user_id = data['user_id']
        
        # =====================================================
        # EDGE CASE 1: Empty String / Whitespace Only
        # =====================================================
        # Mencegah pembuangan compute cost ke LLM jika input kosong
        if not message or not message.strip():
            logger.warning(f"Empty message ditolak dari user {user_id}, session {session_id}")
            return jsonify({
                "error": "Pesan tidak boleh kosong. Silakan tulis keluhan medis Anda."
            }), 400
        
        # Clean whitespace
        message = message.strip()
        
        # =====================================================
        # EDGE CASE 2: Very Long Text (>2000 characters)
        # =====================================================
        # Mencegah overflow pada context window model RAG
        MAX_LENGTH = 2000
        if len(message) > MAX_LENGTH:
            logger.warning(f"Payload Too Large: Message ({len(message)} chars) dari user {user_id}")
            return jsonify({
                "error": "Input terlalu panjang (maksimal 2.000 karakter). Mohon ringkas pertanyaan Anda."
            }), 413
        
        # =====================================================
        # EDGE CASE 3: Code-Switching ID/EN Detection
        # =====================================================
        # Detect jika ada campuran bahasa (untuk analitik kesiapan model multilingual)
        kata_inggris = ['the', 'is', 'are', 'what', 'how', 'disease', 'medicine', 'dose', 'severe', 'headache', 'dizzy', 'feeling']
        kata_indo = ['apa', 'adalah', 'bagaimana', 'penyakit', 'obat', 'dosis', 'gejala', 'sakit', 'kepala', 'lambung', 'buat', 'dok']
        
        has_english = any(word.lower() in kata_inggris for word in message.split())
        has_indonesian = any(word.lower() in kata_indo for word in message.split())
        
        is_code_switching = has_english and has_indonesian
        
        if is_code_switching:
            logger.info(f"Code-switching terdeteksi (ID/EN mix) dari user {user_id}: '{message[:50]}...'")
        
        # =====================================================
        # Validate model_id
        # =====================================================
        model_id = data.get('model_id', 1)
        
        valid_model_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        if model_id not in valid_model_ids:
            return jsonify({
                "error": f"Invalid model_id: {model_id}",
                "valid_model_ids": valid_model_ids,
                "model_info": {
                    "1": "meta-llama/Llama-3.1-8B-Instruct",
                    "2": "Qwen/Qwen2.5-7B-Instruct",
                    "3": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                    "4": "openai/gpt-4o-mini",
                }
            }), 400
        
        # Validate session_id and user_id
        if not isinstance(session_id, int) or session_id <= 0:
            return jsonify({
                "error": "session_id harus integer positif"
            }), 400
        
        if not isinstance(user_id, int) or user_id <= 0:
            return jsonify({
                "error": "user_id harus integer positif"
            }), 400
        
        # =====================================================
        # Process Chat (Dengan perlindungan utuh ke Usecase)
        # =====================================================
        ground_truth = data.get('ground_truth', None)
        
        logger.info(
            f"Chat request medis | User: {user_id} | Session: {session_id} | "
            f"Model: {model_id} | Length: {len(message)} chars"
        )
        
        response_data, status_code = chat_with_history(
            session_id=session_id,
            user_id=user_id,
            user_question=message,
            model_id=model_id,
            ground_truth=ground_truth
        )
        
        # Informasikan di metadata jika menggunakan campuran bahasa
        if is_code_switching:
            response_data['code_switching_detected'] = True
            response_data['code_switching_note'] = (
                "Terdeteksi campuran bahasa Indonesia dan Inggris. "
                "Sistem RAG menggunakan embedding yang mendukung ekstraksi semantik lintas bahasa."
            )
        
        return jsonify(response_data), status_code
        
    except Exception as e:
        logger.error(f"❌ Error dalam handle_chat: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Terjadi error pada sistem: {str(e)}"
        }), 500


# =========================================================================
# FUNGSI MANAJEMEN SESSION (TIDAK DIUBAH, DIPERTAHANKAN AGAR KODE LENGKAP)
# =========================================================================

def handle_create_session():
    data = request.get_json()
    if not data or 'user_id' not in data:
        return jsonify({"error": "Butuh field user_id"}), 400
        
    session_name = data.get('session_name', 'New Chat')
    response_data, status_code = create_new_session(data['user_id'], session_name)
    return jsonify(response_data), status_code

def handle_get_sessions():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "Butuh query parameter user_id"}), 400
        
    response_data, status_code = get_user_sessions(int(user_id))
    return jsonify(response_data), status_code

def handle_get_history(session_id):
    response_data, status_code = get_session_history(session_id)
    return jsonify(response_data), status_code

def handle_update_session(session_id):
    data = request.get_json()
    if not data or 'session_name' not in data:
        return jsonify({"error": "Butuh field session_name"}), 400
        
    response_data, status_code = update_session_name(session_id, data['session_name'])
    return jsonify(response_data), status_code

def handle_delete_session(session_id):
    response_data, status_code = delete_session(session_id)
    return jsonify(response_data), status_code