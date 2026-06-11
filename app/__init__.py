import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()


def create_app():
    """
    Entry point aplikasi Flask - RAG Peraturan Desa.
    Menginisialisasi Flask, CORS, dan registrasi Blueprint routes.
    """
    app = Flask(__name__)

    return app
