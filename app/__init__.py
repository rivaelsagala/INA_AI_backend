import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()


def create_app():
    """
    Entry point aplikasi Flask
    Menginisialisasi Flask, CORS, dan registrasi Blueprint routes.
    """
    app = Flask(__name__)

    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'ina-ai-secret')

    # CORS — izinkan semua origin, expose header custom
    CORS(app, resources={r"/*": {"origins": "*"}}, expose_headers=["X-Chatbot-Text"])

    with app.app_context():
        from app.routes import bp
        app.register_blueprint(bp)
                
    return app
