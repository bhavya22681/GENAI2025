import pyttsx3
import re
import os
import pandas as pd

def text_to_speech(text, filename="static/response.mp3"):
    """
    Convert text into speech and save as an audio file.
    Returns the path to the audio file.
    """
    engine = pyttsx3.init()
    engine.save_to_file(text, filename)
    engine.runAndWait()
    return filename


def clean_text(text: str) -> str:
    """
    Basic text preprocessing:
    - Lowercase
    - Remove special characters
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()


def load_dataset(path: str):
    """
    Safely load dataset as a pandas DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path, encoding="latin1")
    return df

