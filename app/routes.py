from flask import Blueprint

from app.handler.medical_data_handler import (
    handle_ingest_medical_data,
    handle_ingest_custom_medical_file,
    handle_get_medical_data_stats
)

from app.handler.chat_handler import (
    handle_chat,
    handle_create_session,
    handle_get_sessions,
    handle_get_history,
    handle_update_session,
    handle_delete_session
)

bp = Blueprint("routes", __name__)

bp.add_url_rule('/api/ingest-medical-data', 'ingest_medical_data', handle_ingest_medical_data, methods=['POST'])
bp.add_url_rule('/api/ingest-custom-file', 'ingest_custom_file', handle_ingest_custom_medical_file, methods=['POST'])
bp.add_url_rule('/api/medical-data-stats', 'medical_data_stats', handle_get_medical_data_stats, methods=['GET'])


bp.add_url_rule('/api/chat', 'chat', handle_chat, methods=['POST'])
bp.add_url_rule('/api/chat-sessions', 'create_session', handle_create_session, methods=['POST'])
bp.add_url_rule('/api/chat-sessions', 'get_sessions', handle_get_sessions, methods=['GET'])
bp.add_url_rule('/api/chat-history/<int:session_id>', 'get_history', handle_get_history, methods=['GET'])
bp.add_url_rule('/api/chat-sessions/<int:session_id>', 'update_session', handle_update_session, methods=['PUT'])
bp.add_url_rule('/api/chat-sessions/<int:session_id>', 'delete_session', handle_delete_session, methods=['DELETE'])

