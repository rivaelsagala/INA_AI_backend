from typing import Tuple, Dict, Any
from loguru import logger
from app.services.rag_service import get_answer_from_rag
from app.services import chat_history_service

from app.services.guardrails_service import check_medical_guardrails
from app.services.pii_service import has_pii, redact_pii, safe_log

def chat_with_history(session_id: int, user_id: int, user_question: str, model_id: int = 1, ground_truth: str = None) -> Tuple[Dict[str, Any], int]:
    """
    Chat dengan history, guardrails protection, dan PII redaction.
    """
    try:
        # =====================================================
        # 1. GUARDRAILS CHECK (Topical & Intent Routing)
        # =====================================================
        guardrail_result = check_medical_guardrails(user_question)
        
        # Inisialisasi default variable untuk redaction
        pii_detected = False
        pii_summary = {}
        redacted_question = user_question
        
        if guardrail_result["is_blocked"]:
            logger.warning(
                f"BLOCKED by guardrails | User: {user_id} | "
                f"Session: {session_id} | Layer: {guardrail_result['layer']} | "
                f"Category: {guardrail_result['category']} | "
                f"Query: {safe_log(user_question)}"
            )
            
            refusal_answer = guardrail_result["refusal_message"]
            
            # Tetap redact PII sebelum masuk DB (Audit Trail)
            redacted_question, pii_summary = redact_pii(user_question)
            pii_detected = len(pii_summary) > 0
            
            metadata = {
                "sources": [],
                "model_used": "guardrail_system",
                "guardrail_triggered": True,
                "guardrail_category": guardrail_result["category"],
                "guardrail_layer": guardrail_result["layer"],
                "guardrail_reason": guardrail_result["reason"],
                "pii_detected": pii_detected
            }
            chat_history_service.save_chat_message(
                session_id=session_id,
                user_id=user_id,
                user_query=redacted_question, 
                llm_response=refusal_answer,
                metadata=metadata
            )
            
            return {
                "status": "success",
                "message": "Pertanyaan diblokir oleh guardrails",
                "answer": refusal_answer,
                "sources": [],
                "model_used": "guardrail_system",
                "guardrail_triggered": True,
                "guardrail_category": guardrail_result["category"],
                "guardrail_layer": guardrail_result["layer"],
                "pii_detected": pii_detected,
                "pii_warning": "PII detected dan di-redact di log" if pii_detected else None
            }, 200
        
        # =====================================================
        # 2. PII DETECTION & REDACTION (De-identification)
        # =====================================================
        # Lakukan redaksi SEBELUM mengirim prompt ke LLM
        redacted_question, pii_summary = redact_pii(user_question)
        pii_detected = len(pii_summary) > 0
        
        if pii_detected:
            logger.warning(f"PII detected in user input | User: {user_id} | Session: {session_id}")
            logger.info(f"Redacted input to LLM: {redacted_question}")
        else:
            logger.info(f"Processing chat for user {user_id}, session {session_id} with model_id {model_id}")

        # =====================================================
        # 3. RAG PIPELINE
        # =====================================================
        # PENTING: Gunakan redacted_question agar PII tidak masuk ke LLM / API pihak ketiga
        rag_result = get_answer_from_rag(redacted_question, model_id=model_id)
        
        if not rag_result or not rag_result.get("answer"):
            return {"status": "error", "message": "Gagal mendapatkan respons dari sistem RAG"}, 500
        
        # =====================================================
        # 4. FINALIZATION & STORAGE
        # =====================================================
        answer_has_pii = has_pii(rag_result["answer"])
        redacted_answer = rag_result["answer"]
        
        if answer_has_pii:
            logger.warning(f"PII detected in answer! Redacting... | User: {user_id}")
            redacted_answer, _ = redact_pii(rag_result["answer"])
        
        metadata = {
            "sources": rag_result.get("sources", []),
            "model_used": rag_result.get("model_used", "Unknown Model"),
            "guardrail_triggered": False,
            "pii_detected_in_input": pii_detected,
            "pii_detected_in_answer": answer_has_pii,
            "pii_redaction_summary": pii_summary if pii_detected else {}
        }
        
        chat_history_service.save_chat_message(
            session_id=session_id,
            user_id=user_id,
            user_query=redacted_question,
            llm_response=redacted_answer, 
            metadata=metadata
        )
        
        response = {
            "status": "success",
            "message": "Jawaban berhasil diproses",
            "answer": rag_result["answer"],
            "sources": rag_result.get("sources", []),
            "model_used": rag_result.get("model_used", "Unknown Model"),
            "guardrail_triggered": False,
            "pii_detected": pii_detected,
        }
        
        if pii_detected:
            response["pii_warning"] = (
                "Informasi pribadi (seperti Nama/NIK/No.HP) terdeteksi dalam pertanyaan Anda. "
                "Data tersebut telah disamarkan (di-redact) sebelum diproses oleh AI untuk melindungi privasi Anda."
            )
        
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