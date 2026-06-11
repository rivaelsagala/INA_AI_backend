"""
Medical Data Ingestion Handler.

Handler untuk meng-ingest data kesehatan dari CSV/PDF ke vector database.
"""

import os
from flask import request, jsonify
from loguru import logger
from app.usecases.embedding_use_case import ingest_medical_file_to_vector_db

def handle_ingest_medical_data():
    """
    Endpoint untuk meng-ingest medical data dari file default ke vector database.
    """
    try:
        csv_path = os.path.join(
            os.getcwd(), 
            'data', 
            'csv', 
            'daftar_penyakit_dan_dosis_obat.csv'
        )
        
        if not os.path.exists(csv_path):
            logger.error(f"File CSV tidak ditemukan: {csv_path}")
            return jsonify({
                "error": "File CSV medical data tidak ditemukan",
                "path": csv_path
            }), 404
        
        logger.info(f"Mulai ingestion medical data dari: {csv_path}")
        
        result = ingest_medical_file_to_vector_db(csv_path)
        
        status_code = 200 if result["status"] == "success" else 500
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error dalam handle_ingest_medical_data: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Terjadi error: {str(e)}"
        }), 500

def handle_ingest_custom_medical_file():
    """
    Endpoint untuk upload dan ingest custom medical CSV/PDF file.
    """
    if 'file' not in request.files:
        return jsonify({"error": "Key 'file' tidak ditemukan dalam request"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Tidak ada file yang dipilih"}), 400
        
    if not (file.filename.lower().endswith('.csv') or file.filename.lower().endswith('.pdf')):
        return jsonify({"error": "Format file harus CSV atau PDF"}), 400
    
    try:
        from werkzeug.utils import secure_filename
        
        filename = secure_filename(file.filename)
        
        temp_dir = os.path.join(os.getcwd(), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        filepath = os.path.join(temp_dir, filename)
        
        file.save(filepath)
        logger.info(f"File disimpan sementara: {filepath}")
        
        result = ingest_medical_file_to_vector_db(filepath)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"File temporary dihapus: {filepath}")
            
        status_code = 200 if result["status"] == "success" else 500
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error dalam handle_ingest_custom_medical_file: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Terjadi error: {str(e)}"
        }), 500

def handle_get_medical_data_stats():
    """
    Endpoint untuk mendapatkan statistik medical data yang ada di vector database.
    """
    try:
        from app.services.embedding_service import get_medical_data_statistics
        
        stats = get_medical_data_statistics()
        
        return jsonify({
            "status": "success",
            "data": stats
        }), 200
        
    except Exception as e:
        logger.error(f"Error dalam handle_get_medical_data_stats: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Terjadi error: {str(e)}"
        }), 500