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
                    {"role": "system", "content": """Sei un esperto di analisi linguistica.
                    DEVI RISPONDERE SOLO CON UN JSON VALIDO, senza markdown o altro testo.
                    Analizza il testo fornito concentrandoti solo sul contenuto editoriale principale.
                    
                    IGNORA:
                    - Codice e snippet tecnici
                    - Menu di navigazione
                    - Footer e header
                    - Elementi di UI (bottoni, form, etc.)
                    - Messaggi di cookie o privacy
                    - Elementi di social sharing
                    - Meta informazioni (date, autori, tag)
                    
                    ANALIZZA:
                    - Il contenuto principale del testo
                    - Lo stile di scrittura
                    - Il registro linguistico
                    - Le scelte lessicali
                    - I pattern retorici
                    
                    Fornisci l'output in formato JSON con:
                    {
                        "tone": {
                            "formality_score": float,  # 1-10
                            "primary_tone": str,
                            "secondary_tones": list,
                            "characteristic_patterns": list,
                            "notable_expressions": list,
                            "rhetorical_devices": list
                        },
                        "vocabulary": {
                            "complexity_score": float,  # 1-10
                            "domain_specific_terms": list,
                            "recurring_phrases": list,
                            "register_level": str,
                            "distinctive_word_choices": list,
                            "semantic_fields": list
                        }
                    }"""},
                    {"role": "user", "content": cleaned_text[:4000]}
                ],
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content.strip()
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            try:
                return json.loads(response_text)
            except json.JSONDecodeError as je:
                print(f"Errore nel parsing JSON per il documento: {je}")
                print("Risposta che ha causato l'errore:")
                print(response_text)
                return None
                
        except Exception as e:
            print(f"Errore nell'analisi del documento: {e}")
            return None

    def generate_markdown_reports(self, analyses: list) -> tuple:
        """Genera report markdown con gestione analisi parziali"""
        successful_analyses = [a for a in analyses if a is not None]
        failed_analyses = len(analyses) - len(successful_analyses)
        
        if not successful_analyses:
            print("Nessuna analisi completata con successo.")
            return None, None
            
        tone_scores = [a['tone']['formality_score'] for a in successful_analyses]
        vocab_scores = [a['vocabulary']['complexity_score'] for a in successful_analyses]
        
        stats = {
            'tone': {
                'median': np.median(tone_scores),
                'mean': np.mean(tone_scores),
                'std': np.std(tone_scores),
                'q1': np.percentile(tone_scores, 25),
                'q3': np.percentile(tone_scores, 75)
            },
            'vocab': {
                'median': np.median(vocab_scores),
                'mean': np.mean(vocab_scores),
                'std': np.std(vocab_scores),
                'q1': np.percentile(vocab_scores, 25),
                'q3': np.percentile(vocab_scores, 75)
            }
        }
        
        patterns = {
            'tone': {
                'primary': Counter([a['tone']['primary_tone'] for a in successful_analyses]).most_common(),
                'rhetorical': Counter([d for a in successful_analyses for d in a['tone']['rhetorical_devices']]).most_common(),
                'expressions': Counter([e for a in successful_analyses for e in a['tone']['notable_expressions']]).most_common()
            },
            'vocab': {
                'terms': Counter([t for a in successful_analyses for t in a['vocabulary']['domain_specific_terms']]).most_common(),
                'phrases': Counter([p for a in successful_analyses for p in a['vocabulary']['recurring_phrases']]).most_common(),
                'fields': Counter([f for a in successful_analyses for f in a['vocabulary']['semantic_fields']]).most_common()
            }
        }
        
        tone_prompt = f"""
        Genera un report dettagliato (minimo 1000 parole) sul tone of voice.
        NOTA: {failed_analyses} documenti su {len(analyses)} non sono stati analizzati a causa di errori.
        Questa analisi si basa sui {len(successful_analyses)} documenti analizzati con successo.
        
        Statistiche di formalità:
        {json.dumps(stats['tone'], indent=2)}
        
        Pattern identificati:
        {json.dumps(patterns['tone'], indent=2)}
        
        Il report deve includere:
        1. Executive Summary
        2. Analisi Statistica Dettagliata
        3. Pattern Stilistici Identificati
        4. Analisi dei Dispositivi Retorici
        5. Variazioni e Casi Particolari
        6. Implicazioni e Best Practices
        7. Raccomandazioni per la Coerenza
        8. Esempi Concreti e Analisi
        9. Conclusioni e Next Steps
        
        Ogni sezione deve essere approfondita e supportata da esempi specifici.
        """
        
        vocab_prompt = f"""
        Genera un report dettagliato (minimo 1000 parole) sul vocabulary.
        NOTA: {failed_analyses} documenti su {len(analyses)} non sono stati analizzati a causa di errori.
        Questa analisi si basa sui {len(successful_analyses)} documenti analizzati con successo.
        
        Statistiche di complessità:
        {json.dumps(stats['vocab'], indent=2)}
        
        Pattern identificati:
        {json.dumps(patterns['vocab'], indent=2)}
        
        Il report deve includere:
        1. Executive Summary
        2. Analisi Statistica Dettagliata
        3. Campi Semantici Dominanti
        4. Pattern Lessicali
        5. Analisi del Registro Linguistico
        6. Termini Specialistici e Loro Uso
        7. Variazioni e Tendenze
        8. Best Practices e Linee Guida
        9. Esempi e Casi Studio
        10. Conclusioni e Raccomandazioni
        
        Ogni sezione deve essere approfondita e supportata da esempi specifici.
        """
        
        tone_response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Sei un esperto di analisi linguistica specializzato in tone of voice."},
                {"role": "user", "content": tone_prompt}
            ],
            temperature=0.7
        )
        
        vocab_response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Sei un esperto di analisi lessicale e linguistica computazionale."},
                {"role": "user", "content": vocab_prompt}
            ],
            temperature=0.7
        )
        
        return tone_response.choices[0].message.content, vocab_response.choices[0].message.content

    def validate_report_length(self, text: str, min_words: int = 1000) -> bool:
        """Verifica che il report abbia almeno il numero minimo di parole richiesto."""
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
