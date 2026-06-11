from app.services.preprocessing_service import create_medical_documents
from app.services.embedding_service import store_medical_documents_to_supabase, get_medical_data_statistics
from loguru import logger

def ingest_medical_file_to_vector_db(file_path: str):
    """
    Ingest medical data dari CSV atau dokumen PDF ke vector database.
    
    Args:
        file_path: Path ke file data
        
    Returns:
        Dictionary dengan status dan message
    """
    try:
        logger.info(f"Mulai processing medical file: {file_path}")
        
        # Langkah 1: Buat documents dari CSV atau PDF (sudah di-chunk)
        documents = create_medical_documents(file_path)
        
        if not documents:
            return {
                "status": "error",
                "message": "Tidak ada data yang berhasil diproses dari file"
            }
        
        total_chunks = len(documents)
        logger.info(f"Statistik: {total_chunks} chunk documents siap disimpan")
        
        # Langkah 2: Store ke vector database
        logger.info("Menyimpan documents ke vector database...")
        store_medical_documents_to_supabase(documents)
        
        # Langkah 3: Dapatkan statistik keseluruhan terbaru dari DB
        stats = get_medical_data_statistics()
        
        return {
            "status": "success",
            "message": f"Berhasil meng-ingest {total_chunks} medical chunks ke database",
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Error dalam ingest_medical_file_to_vector_db: {str(e)}")
        return {
            "status": "error",
            "message": f"Gagal memproses medical data: {str(e)}"
        }

# Legacy function untuk backward compatibility (deprecated)
def ingest_pdf_to_vector_db(file_path: str, original_filename: str):
    """
    DEPRECATED: Gunakan ingest_medical_file_to_vector_db untuk medical chatbot.
    """
    logger.warning("ingest_pdf_to_vector_db is deprecated. Use ingest_medical_file_to_vector_db instead.")
    return ingest_medical_file_to_vector_db(file_path)