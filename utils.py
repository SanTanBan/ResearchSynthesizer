import os
from docx import Document
import logging

def load_together_api_key():
    """Load Together AI API key from the internal Word document"""
    try:
        doc = Document("api_key.docx")
        api_key = " ".join(paragraph.text for paragraph in doc.paragraphs).strip()
        if api_key and not api_key.startswith("YOUR_"):
            return api_key
        logging.warning("API key not properly set in api_key.docx")
        return None
    except Exception as e:
        logging.error(f"Error loading API key: {str(e)}")
        return None
