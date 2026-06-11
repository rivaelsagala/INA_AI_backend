from flask import Blueprint

from app.handler.medical_data_handler import (
    handle_ingest_medical_data,
    handle_ingest_custom_medical_file,
    handle_get_medical_data_stats
)

bp = Blueprint("routes", __name__)

bp.add_url_rule('/api/ingest-medical-data', 'ingest_medical_data', handle_ingest_medical_data, methods=['POST'])
bp.add_url_rule('/api/ingest-custom-file', 'ingest_custom_file', handle_ingest_custom_medical_file, methods=['POST'])
bp.add_url_rule('/api/medical-data-stats', 'medical_data_stats', handle_get_medical_data_stats, methods=['GET'])
