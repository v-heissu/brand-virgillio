import os
import json
import re
import io
import numpy as np
import nltk
from collections import Counter
import openai
import docx
import PyPDF2
from pathlib import Path
from nltk.tokenize import word_tokenize
from text_cleaner import TextCleaner

class DocumentProcessor:
    def __init__(self, api_key=None):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OpenAI API key non trovata")
        
        self.client = openai.OpenAI(api_key=api_key)
        
    def analyze_document(self, text: str) -> dict:
        """Analizza un singolo documento per tone e vocabulary"""
        cleaned_text = TextCleaner.clean_text(text)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Analisi linguistica (JSON output)"},
                    {"role": "user", "content": cleaned_text[:4000]}
                ],
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content.strip()
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                print("Errore nel parsing JSON")
                return None
                
        except Exception as e:
            print(f"Errore nell'analisi del documento: {e}")
            return None

    def generate_markdown_reports(self, analyses: list) -> tuple:
        """Genera report markdown"""
        # Implementazione simile al codice originale
        # Per brevità, ho omesso alcuni dettagli
        pass

    def validate_report_length(self, text: str, min_words: int = 1000) -> bool:
        """Verifica la lunghezza del report"""
        words = word_tokenize(text)
        return len(words) >= min_words

def read_document(file_path):
    """Legge documenti di diversi formati"""
    ext = Path(file_path).suffix.lower()
    
    try:
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.docx':
            doc = docx.Document(file_path)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        elif ext == '.pdf':
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                return '\n'.join([page.extract_text() for page in pdf_reader.pages])
        else:
            raise ValueError(f"Formato file {ext} non supportato")
    except Exception as e:
        print(f"Errore nel leggere {file_path}: {e}")
        return None