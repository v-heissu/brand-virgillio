import re
from bs4 import BeautifulSoup

class TextCleaner:
    """Classe per la pulizia e pre-processing del testo."""
    
    @staticmethod
    def clean_html(text):
        """Rimuove tag HTML e contenuti non rilevanti."""
        soup = BeautifulSoup(text, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.decompose()
            
        for element in soup(["nav", "header", "footer", "button", "aside"]):
            element.decompose()
            
        return soup.get_text()
    
    @staticmethod
    def clean_code_blocks(text):
        """Rimuove blocchi di codice."""
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`.*?`', '', text)
        text = re.sub(r'<code>[\s\S]*?</code>', '', text)
        text = re.sub(r'<pre>[\s\S]*?</pre>', '', text)
        return text
    
    @staticmethod
    def clean_urls(text):
        """Rimuove URL."""
        return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    @staticmethod
    def clean_common_web_elements(text):
        """Rimuove elementi comuni delle pagine web."""
        patterns = [
            r'Accept (?:all )?cookies?',
            r'Cookie Policy',
            r'Privacy Policy',
            r'Terms (?:of|and) (?:Use|Service)',
            r'Subscribe to our newsletter',
            r'Sign up',
            r'Log in',
            r'Menu',
            r'Search',
            r'Close',
            r'\[advertisement\]',
            r'Share this',
            r'Follow us',
            r'©.*?(?=\n|\r|$)',
            r'\[Read More\]',
            r'\[Click here\]',
            r'\[\.\.\.\]'
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text
    
    @staticmethod
    def normalize_whitespace(text):
        """Normalizza gli spazi bianchi."""
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()
    
    @staticmethod
    def clean_text(text):
        """Applica tutte le pulizie in sequenza."""
        text = TextCleaner.clean_html(text)
        text = TextCleaner.clean_code_blocks(text)
        text = TextCleaner.clean_urls(text)
        text = TextCleaner.clean_common_web_elements(text)
        text = TextCleaner.normalize_whitespace(text)
        return text