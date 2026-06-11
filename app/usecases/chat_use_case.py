from typing import Tuple, Dict, Any
from loguru import logger
from app.services.rag_service import get_answer_from_rag
from app.services import chat_history_service
# from app.services.ragas_service import ragas_service          # DISABLED: Belum digunakan
# from app.services.guardrails_service import check_medical_guardrails  # DISABLED: Belum digunakan
# from app.services.pii_service import has_pii, redact_pii, safe_log   # DISABLED: Belum digunakan

def chat_with_history(session_id: int, user_id: int, user_question: str, model_id: int = 1, ground_truth: str = None) -> Tuple[Dict[str, Any], int]:
    """
    Chat dengan history.
    Fitur guardrails, PII redaction, dan RAGAS evaluation sementara dinonaktifkan.
    """
    try:
        logger.info(f"Processing chat for user {user_id}, session {session_id} with model_id {model_id}")

        # =====================================================
        # RAG PIPELINE (langsung tanpa guardrails & PII)
        # =====================================================
        rag_result = get_answer_from_rag(user_question, model_id=model_id)
        
        if not rag_result or not rag_result.get("answer"):
            return {"status": "error", "message": "Gagal mendapatkan respons dari sistem RAG"}, 500
        
        # =====================================================
        # SIMPAN KE HISTORY
        # =====================================================
        metadata = {
            "sources": rag_result.get("sources", []),
            "model_used": rag_result.get("model_used", "Unknown Model"),
        }
        
        chat_history_service.save_chat_message(
            session_id=session_id,
            user_id=user_id,
            user_query=user_question,
            llm_response=rag_result["answer"], 
            metadata=metadata
        )
        
        response = {
            "status": "success",
            "message": "Jawaban berhasil diproses",
            "answer": rag_result["answer"],
            "sources": rag_result.get("sources", []),
            "model_used": rag_result.get("model_used", "Unknown Model"),
        }
        
        return response, 200

    except Exception as e:
        logger.error(f"Error in chat_with_history: {str(e)}")
        return {"status": "error", "message": f"Server error: {str(e)}"}, 500

# (Fungsi create_new_session, get_user_sessions, dll di bawahnya tetap sama seperti aslinya)
def create_new_session(user_id: int, session_name: str) -> Tuple[Dict[str, Any], int]:
    session_id = chat_history_service.create_chat_session(user_id, session_name)
    if session_id:
        return {"status": "success", "data": {"id": session_id, "session_name": session_name}}, 200
    return {"status": "error", "message": "Gagal membuat session"}, 500

def get_user_sessions(user_id: int) -> Tuple[Dict[str, Any], int]:
    sessions = chat_history_service.get_user_sessions(user_id)
    return {"status": "success", "data": sessions}, 200

def get_session_history(session_id: int) -> Tuple[Dict[str, Any], int]:
    history = chat_history_service.get_session_history(session_id)
    return {"status": "success", "data": history}, 200

def update_session_name(session_id: int, new_name: str) -> Tuple[Dict[str, Any], int]:
    success = chat_history_service.update_session_name(session_id, new_name)
    if success:
        return {"status": "success", "message": "Nama session berhasil diupdate"}, 200
    return {"status": "error", "message": "Gagal mengupdate session"}, 500

def delete_session(session_id: int) -> Tuple[Dict[str, Any], int]:
    success = chat_history_service.delete_session(session_id)
    if success:
        return {"status": "success", "message": "Session berhasil dihapus"}, 200
    return {"status": "error", "message": "Gagal menghapus session"}, 500