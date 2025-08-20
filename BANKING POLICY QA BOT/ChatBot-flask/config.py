import os

class Config:
    # Flask secret key (needed for sessions, forms, etc.)
    SECRET_KEY = os.environ.get("SECRET_KEY") or "supersecretkey"

    # Debug mode
    DEBUG = True

    # Dataset path
    DATASET_PATH = os.path.join(
        os.path.dirname(__file__), 
        "data", 
        "archive2", 
        "Dataset_Banking_chatbot.csv"
    )

    # Static folder for audio responses
    STATIC_AUDIO_PATH = os.path.join("static", "audio")

    # Text-to-Speech settings
    TTS_LANGUAGE = "en"
    TTS_SLOW = False
