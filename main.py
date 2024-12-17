import os
import streamlit as st
from document_processor import DocumentProcessor, read_document

def main():
    st.title("Analisi Linguistica dei Documenti")
    
    # Configurazione API Key
    st.sidebar.header("Configurazione OpenAI")
    api_key = st.sidebar.text_input("Inserisci OpenAI API Key", type="password")
    
    # Upload documenti
    uploaded_files = st.file_uploader(
        "Carica documenti (txt, docx, pdf)", 
        type=['txt', 'docx', 'pdf'], 
        accept_multiple_files=True
    )
    
    if uploaded_files and api_key:
        try:
            processor = DocumentProcessor(api_key)
            analyses = []
            
            for uploaded_file in uploaded_files:
                # Salva temporaneamente il file
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                text = read_document(uploaded_file.name)
                
                if text:
                    analysis = processor.analyze_document(text)
                    if analysis:
                        analyses.append(analysis)
                
                # Rimuovi il file temporaneo
                os.remove(uploaded_file.name)
            
            if analyses:
                tone_report, vocab_report = processor.generate_markdown_reports(analyses)
                
                st.success("Analisi completata!")
                
                # Display reports
                st.subheader("Report Tone of Voice")
                st.markdown(tone_report)
                
                st.subheader("Report Vocabulary")
                st.markdown(vocab_report)
                
                # Download reports
                st.download_button(
                    label="Scarica Report Tone",
                    data=tone_report,
                    file_name="tone_analysis.md",
                    mime="text/markdown"
                )
                
                st.download_button(
                    label="Scarica Report Vocabulary",
                    data=vocab_report,
                    file_name="vocabulary_analysis.md",
                    mime="text/markdown"
                )
            
        except Exception as e:
            st.error(f"Errore durante l'analisi: {e}")
    
    elif uploaded_files and not api_key:
        st.warning("Per favore inserisci l'API key di OpenAI")

if __name__ == "__main__":
    main()