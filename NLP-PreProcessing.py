"""
Nama: Patrick Andrasena Tumengkol
Mata Kuliah: RKK305 Pemrosesan Bahasa Alami
Tugas: Analisis Dokumen Teks Berita Bahasa Indonesia

Complete implementation with all necessary file generation components:
1. Regex untuk pembersihan pola teks
2. Preprocessing lengkap (tokenisasi, stopword removal, stemming/lemmatization)
3. Representasi fitur dengan TF, IDF, TF-IDF
4. POS tagging, NER, dan sequence labeling dasar
5. Model n-gram untuk prediksi kata berikutnya
6. Complete file generation for reports (plots, Excel, HTML, etc.)
"""

import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
import time
import random
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import feedparser
from datetime import datetime
from urllib.parse import urljoin, urlparse
import json
import os
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

warnings.filterwarnings('ignore')

# Set matplotlib backend for better compatibility
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# Create output directory
os.makedirs('indonesian_news_analysis_output', exist_ok=True)

# Indonesian NLP libraries
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.tag import pos_tag
    import spacy
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    SASTRAWI_AVAILABLE = True
except ImportError as e:
    print("Warning: Some packages not available:", e)
    print("Please install required packages:")
    print("pip install nltk spacy sastrawi scikit-learn matplotlib seaborn pandas numpy")
    print("pip install requests beautifulsoup4 feedparser wordcloud plotly openpyxl")
    print("python -m spacy download en_core_web_sm")
    StemmerFactory = None
    StopWordRemoverFactory = None
    SASTRAWI_AVAILABLE = False

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
except:
    pass

class IndonesianNewsAnalyzer:
    def __init__(self):
        """Initialize Indonesian News Text Analysis System"""
        self.setup_indonesian_tools()
        self.articles = []
        self.processed_articles = []
        
    def setup_indonesian_tools(self):
        """Setup Indonesian-specific NLP tools"""
        # Comprehensive Indonesian stopwords
        self.indonesian_stopwords = set([
            'yang', 'dan', 'di', 'ke', 'dari', 'dalam', 'untuk', 'pada', 'dengan', 'oleh',
            'adalah', 'akan', 'telah', 'sudah', 'dapat', 'bisa', 'juga', 'ini', 'itu',
            'atau', 'jika', 'karena', 'sehingga', 'namun', 'tetapi', 'maka', 'agar',
            'bahwa', 'sebagai', 'antara', 'atas', 'bawah', 'kiri', 'kanan', 'depan',
            'belakang', 'samping', 'luar', 'dalam', 'kami', 'kita', 'mereka', 'dia',
            'ia', 'nya', 'mu', 'ku', 'saya', 'anda', 'kamu', 'beliau'
        ])
        
        # Setup Indonesian stemmer and stopword remover if available
        if SASTRAWI_AVAILABLE and StemmerFactory is not None:
            try:
                # Indonesian stemmer
                factory = StemmerFactory()
                self.stemmer = factory.create_stemmer()
                
                # Indonesian stopwords
                stopword_factory = StopWordRemoverFactory()
                self.stopword_remover = stopword_factory.create_stop_word_remover()
                
                print("✅ Sastrawi Indonesian NLP tools loaded successfully")
                
            except Exception as e:
                print(f"Warning: Error initializing Sastrawi tools: {e}")
                self.stemmer = None
                self.stopword_remover = None
        else:
            print("Warning: Sastrawi library not available. Using basic Indonesian processing.")
            self.stemmer = None
            self.stopword_remover = None
            
        # Try to load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")  # Will be used for basic NER patterns
            print("✅ spaCy model loaded successfully")
        except Exception as e:
            print(f"Warning: spaCy model not found: {e}. NER will use basic patterns.")
            self.nlp = None

    def get_user_agent(self) -> str:
        """Return a random user agent to avoid blocking"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        return random.choice(user_agents)
    
    def fetch_rss_feed(self, rss_url: str, source_name: str) -> List[Dict]:
        """Fetch articles from RSS feed"""
        articles = []
        try:
            # Set user agent to avoid blocking
            headers = {'User-Agent': self.get_user_agent()}
            
            # Parse RSS feed
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries:
                # Parse published date
                try:
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_date = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        published_date = datetime(*entry.updated_parsed[:6])
                    else:
                        published_date = datetime.now()
                except:
                    published_date = datetime.now()
                
                # Get content (try different fields)
                content = ""
                if hasattr(entry, 'content') and entry.content:
                    content = entry.content[0].value if isinstance(entry.content, list) else entry.content
                elif hasattr(entry, 'summary'):
                    content = entry.summary
                elif hasattr(entry, 'description'):
                    content = entry.description
                
                # Clean HTML from content
                if content:
                    soup = BeautifulSoup(content, 'html.parser')
                    content = soup.get_text().strip()
                
                article = {
                    'title': entry.title if hasattr(entry, 'title') else 'No Title',
                    'content': content,
                    'url': entry.link if hasattr(entry, 'link') else '',
                    'published_date': published_date,
                    'source': source_name
                }
                articles.append(article)
                
        except Exception as e:
            print(f"Error fetching RSS from {rss_url}: {e}")
        
        return articles
    
    def scrape_article_content(self, url: str) -> str:
        """Scrape full article content from URL"""
        try:
            headers = {'User-Agent': self.get_user_agent()}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                    script.decompose()
                
                # Try different content selectors
                content_selectors = [
                    '.post-content', '.article-content', '.content', 'article',
                    '.entry-content', 'main', '[role="main"]', '.post-body'
                ]
                
                content = ""
                for selector in content_selectors:
                    content_element = soup.select_one(selector)
                    if content_element:
                        paragraphs = content_element.find_all('p')
                        if paragraphs:
                            content = ' '.join([p.get_text().strip() for p in paragraphs])
                            break
                
                # Fallback: get all paragraph text
                if not content:
                    all_paragraphs = soup.find_all('p')
                    content = ' '.join([p.get_text().strip() for p in all_paragraphs])
                
                return content[:5000]  # Limit content length
                
        except Exception as e:
            print(f"Error scraping {url}: {e}")
        
        return ""

    def clean_text_with_regex(self, text: str) -> str:
        """Enhanced text cleaning using Regular Expression"""
        if not text:
            return ""
        
        # Remove HTML tags and entities
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'&[a-zA-Z0-9]+;', '', text)
        
        # Remove URLs and emails
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        # Remove phone numbers (Indonesian format)
        text = re.sub(r'\b(?:\+62|0)\d{2,4}[-\s]?\d{3,4}[-\s]?\d{3,4}\b', '', text)
        
        # Remove standalone numbers but keep currency and dates
        text = re.sub(r'\b\d+\b(?!\s*(?:rupiah|ribu|juta|miliar|Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember))', '', text)
        
        # Clean quotation marks and special characters
        text = re.sub(r'["""''`]', '"', text)
        text = re.sub(r'[^\w\s\.,;:!?\-"()]', ' ', text)
        
        # Clean multiple spaces and punctuation
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\n\r\t]+', ' ', text)
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[,]{2,}', ',', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text.strip()

    def preprocess_text(self, text: str) -> Dict:
        """Enhanced text preprocessing for Indonesian text"""
        if not text:
            return {"original": "", "cleaned": "", "tokens": [], "no_stopwords": [], "stemmed": [], "filtered": []}
        
        # 1. Clean text with regex
        cleaned_text = self.clean_text_with_regex(text)
        
        # 2. Tokenization
        tokens = self._enhanced_tokenize(cleaned_text.lower())
        
        # 3. Filter tokens
        filtered_tokens = [token for token in tokens if len(token) > 2 and not token.isdigit() and token.isalpha()]
        
        # 4. Remove stopwords
        if self.stopword_remover:
            no_stopwords_text = self.stopword_remover.remove(' '.join(filtered_tokens))
            no_stopwords = word_tokenize(no_stopwords_text)
        else:
            no_stopwords = [token for token in filtered_tokens if token not in self.indonesian_stopwords]
        
        # 5. Stemming
        if self.stemmer:
            stemmed = [self.stemmer.stem(token) for token in no_stopwords]
        else:
            stemmed = [self._basic_stem(token) for token in no_stopwords]
        
        # 6. Final filtering
        stemmed = [word for word in stemmed if len(word) > 2 and word not in {'ada', 'jadi', 'buat', 'kasi'}]
        
        return {
            "original": text,
            "cleaned": cleaned_text,
            "tokens": tokens,
            "filtered": filtered_tokens,
            "no_stopwords": no_stopwords,
            "stemmed": stemmed
        }
    
    def _enhanced_tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization handling Indonesian contractions"""
        # Handle Indonesian contractions
        text = re.sub(r'\btidak\s+bisa\b', 'tidak_bisa', text)
        text = re.sub(r'\btidak\s+dapat\b', 'tidak_dapat', text)
        text = re.sub(r'\bsudah\s+tidak\b', 'sudah_tidak', text)
        text = re.sub(r'\bbelum\s+bisa\b', 'belum_bisa', text)
        
        tokens = word_tokenize(text)
        
        # Post-process tokens
        processed_tokens = []
        for token in tokens:
            if '_' in token:
                processed_tokens.extend(token.split('_'))
            else:
                processed_tokens.append(token)
        
        return processed_tokens
    
    def _basic_stem(self, word: str) -> str:
        """Basic Indonesian stemming for fallback"""
        # Remove common suffixes
        suffixes = ['nya', 'kan', 'an', 'i']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 3:
                word = word[:-len(suffix)]
                break
        
        # Remove common prefixes
        prefixes = ['ber', 'ter', 'me', 'mem', 'men', 'meng', 'meny', 'pe', 'per', 'di', 'ke']
        for prefix in prefixes:
            if word.startswith(prefix) and len(word) > len(prefix) + 3:
                word = word[len(prefix):]
                break
        
        return word

    def scrape_indonesian_news(self, num_articles: int = 30) -> List[Dict]:
        """Scrape Indonesian news articles from multiple sources"""
        articles = []
        
        # Multiple Indonesian RSS feed URLs
        rss_feeds = [
            ("https://www.antaranews.com/rss/terkini.xml", "Antara News"),
            ("https://www.antaranews.com/rss/politik.xml", "Antara News - Politik"),
            ("https://www.antaranews.com/rss/ekonomi.xml", "Antara News - Ekonomi"),
            ("https://www.antaranews.com/rss/tekno.xml", "Antara News - Teknologi"),
        ]
        
        print("Mengambil artikel dari berbagai sumber berita Indonesia...")
        
        # Fetch articles from RSS feeds
        for rss_url, source_name in rss_feeds:
            if len(articles) >= num_articles:
                break
                
            try:
                feed_articles = self.fetch_rss_feed(rss_url, source_name)
                
                for article in feed_articles:
                    if len(articles) >= num_articles:
                        break
                    
                    # Try to get full content
                    if article['url'] and article['content']:
                        full_content = self.scrape_article_content(article['url'])
                        if full_content:
                            article['content'] = full_content
                    
                    # Only add if we have meaningful content
                    if article['content'] and len(article['content'].strip()) > 50:
                        articles.append(article)
                        
                time.sleep(1)  # Be respectful to servers
                
            except Exception as e:
                print(f"Error fetching from {rss_url}: {e}")
                continue
        
        # Add sample articles if needed
        if len(articles) < num_articles:
            print("Menambahkan artikel contoh untuk melengkapi dataset...")
            sample_articles = self._get_sample_articles()
            
            while len(articles) < num_articles:
                for article in sample_articles:
                    if len(articles) >= num_articles:
                        break
                    variation_id = len(articles) + 1
                    articles.append({
                        **article,
                        "id": variation_id,
                        "title": f"{article['title']} (Artikel {variation_id})",
                        "content": article['content']
                    })
        
        print(f"Berhasil mengumpulkan {len(articles)} artikel")
        return articles[:num_articles]

    def _get_sample_articles(self) -> List[Dict]:
        """Get sample Indonesian news articles"""
        return [
            {
                "title": "Pemerintah Indonesia Luncurkan Program Digitalisasi UMKM",
                "content": "Pemerintah Indonesia telah meluncurkan program digitalisasi untuk Usaha Mikro Kecil dan Menengah (UMKM). Program ini bertujuan untuk meningkatkan daya saing UMKM di era digital. Menteri Koperasi dan UKM menyatakan bahwa program ini akan memberikan akses teknologi dan pelatihan digital kepada para pelaku UMKM di seluruh Indonesia. Program digitalisasi ini meliputi pelatihan penggunaan platform e-commerce, digital marketing, dan sistem manajemen keuangan digital. Pemerintah juga menyediakan bantuan dana untuk memfasilitasi transformasi digital UMKM. Inisiatif ini diharapkan dapat meningkatkan produktivitas dan jangkauan pasar UMKM Indonesia.",
                "url": "https://example.com/article1",
                "source": "Sample Data",
                "published_date": datetime.now()
            },
            {
                "title": "Teknologi Artificial Intelligence Mulai Diterapkan di Sektor Kesehatan",
                "content": "Rumah sakit di Indonesia mulai menerapkan teknologi kecerdasan buatan untuk meningkatkan layanan kesehatan. Sistem AI digunakan untuk diagnosis medis dan analisis data pasien. Dokter spesialis menyambut baik inovasi ini karena dapat membantu meningkatkan akurasi diagnosis dan efisiensi pelayanan. Beberapa rumah sakit telah menggunakan AI untuk mendeteksi kanker, analisis radiologi, dan prediksi risiko penyakit. Teknologi ini juga membantu dalam manajemen jadwal operasi dan optimalisasi sumber daya medis. Implementasi AI dalam kesehatan diharapkan dapat mengurangi kesalahan medis dan meningkatkan kualitas pelayanan kesehatan di Indonesia.",
                "url": "https://example.com/article2",
                "source": "Sample Data",
                "published_date": datetime.now()
            },
            {
                "title": "Pendidikan Digital Menjadi Prioritas di Era Pandemi",
                "content": "Kementerian Pendidikan Indonesia mengumumkan bahwa pendidikan digital menjadi prioritas utama dalam sistem pembelajaran. Platform e-learning dikembangkan untuk mendukung pembelajaran jarak jauh. Para guru dan siswa diberikan pelatihan untuk menggunakan teknologi pendidikan modern. Sistem pembelajaran hybrid yang menggabungkan online dan offline mulai diterapkan di berbagai sekolah. Pemerintah juga menyediakan perangkat komputer dan akses internet untuk mendukung pembelajaran digital di daerah terpencil. Transformasi digital dalam pendidikan ini bertujuan untuk memastikan kontinuitas pembelajaran dan meningkatkan kualitas pendidikan di Indonesia.",
                "url": "https://example.com/article3",
                "source": "Sample Data",
                "published_date": datetime.now()
            }
        ]

    def calculate_tf_idf(self, documents: List[str]) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
        """Calculate TF, IDF, and TF-IDF matrices manually"""
        # Create vocabulary
        vocabulary = set()
        for doc in documents:
            vocabulary.update(doc.split())
        vocabulary = sorted(list(vocabulary))
        
        # Initialize matrices
        tf_matrix = np.zeros((len(documents), len(vocabulary)))
        
        # Calculate TF (Term Frequency)
        for doc_idx, doc in enumerate(documents):
            words = doc.split()
            word_count = len(words)
            word_freq = Counter(words)
            
            for word_idx, word in enumerate(vocabulary):
                if word_count > 0:
                    tf_matrix[doc_idx, word_idx] = word_freq[word] / word_count
        
        # Calculate IDF (Inverse Document Frequency)
        doc_count = len(documents)
        idf_array = np.zeros(len(vocabulary))
        
        for word_idx, word in enumerate(vocabulary):
            docs_with_word = sum(1 for doc in documents if word in doc.split())
            if docs_with_word > 0:
                idf_array[word_idx] = np.log(doc_count / docs_with_word)
        
        # Calculate TF-IDF
        tf_idf_matrix = tf_matrix * idf_array
        
        return tf_idf_matrix, vocabulary, tf_matrix, idf_array

    def get_top_words_per_document(self, tf_idf_matrix: np.ndarray, vocabulary: List[str], top_k: int = 10) -> List[List[Tuple]]:
        """Get top k words with highest TF-IDF scores for each document"""
        top_words_per_doc = []
        
        for doc_idx in range(tf_idf_matrix.shape[0]):
            doc_scores = tf_idf_matrix[doc_idx]
            top_indices = np.argsort(doc_scores)[::-1][:top_k]
            top_words = [(vocabulary[idx], doc_scores[idx]) for idx in top_indices if doc_scores[idx] > 0]
            top_words_per_doc.append(top_words)
        
        return top_words_per_doc

    def pos_tagging_indonesian(self, text: str) -> List[Tuple]:
        """Enhanced POS tagging for Indonesian text"""
        tokens = word_tokenize(text.lower())
        pos_tags = []
        for token in tokens:
            tag = self._determine_pos_tag(token)
            pos_tags.append((token, tag))
        return pos_tags
    
    def _determine_pos_tag(self, token: str) -> str:
        """Determine POS tag for Indonesian token"""
        # Comprehensive POS tagging rules for Indonesian
        conjunctions = {'dan', 'atau', 'tetapi', 'namun', 'serta', 'bahkan', 'lalu', 'kemudian'}
        if token in conjunctions:
            return 'CC'
        
        prepositions = {'di', 'ke', 'dari', 'pada', 'dalam', 'oleh', 'untuk', 'dengan', 'terhadap'}
        if token in prepositions:
            return 'IN'
        
        determiners = {'yang', 'ini', 'itu', 'tersebut', 'para', 'sang', 'si', 'sebuah', 'seorang'}
        if token in determiners:
            return 'DT'
        
        pronouns = {'saya', 'aku', 'kamu', 'anda', 'dia', 'ia', 'mereka', 'kami', 'kita', 'beliau'}
        if token in pronouns or token.endswith('nya'):
            return 'PRP'
        
        auxiliaries = {'adalah', 'akan', 'sedang', 'telah', 'sudah', 'dapat', 'bisa', 'mampu'}
        if token in auxiliaries:
            return 'MD'
        
        if self._is_verb(token):
            return 'VB'
        
        if self._is_adjective(token):
            return 'JJ'
        
        adverbs = {'sangat', 'amat', 'sekali', 'terlalu', 'cukup', 'selalu', 'sering', 'tidak', 'belum'}
        if token in adverbs:
            return 'RB'
        
        if token.isdigit() or self._is_indonesian_number(token):
            return 'CD'
        
        return 'NN'  # Default to noun
    
    def _is_verb(self, token: str) -> bool:
        """Check if token follows Indonesian verb patterns"""
        verb_prefixes = ['ber', 'me', 'mem', 'men', 'meng', 'meny', 'ter', 'di']
        verb_suffixes = ['kan', 'i', 'an']
        
        for prefix in verb_prefixes:
            if token.startswith(prefix) and len(token) > len(prefix) + 2:
                return True
        
        for suffix in verb_suffixes:
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                return True
        
        base_verbs = {'makan', 'minum', 'tidur', 'bangun', 'pergi', 'datang', 'lihat', 'dengar', 'bicara', 'kata', 'tulis', 'baca', 'kerja', 'main', 'belajar', 'ajar'}
        return token in base_verbs
    
    def _is_adjective(self, token: str) -> bool:
        """Check if token follows Indonesian adjective patterns"""
        common_adjectives = {'besar', 'kecil', 'tinggi', 'rendah', 'panjang', 'pendek', 'luas', 'sempit', 'baik', 'buruk', 'bagus', 'jelek', 'cantik', 'indah', 'baru', 'lama', 'muda', 'tua', 'mudah', 'sulit', 'cepat', 'lambat', 'kuat', 'lemah'}
        return token in common_adjectives
    
    def _is_indonesian_number(self, token: str) -> bool:
        """Check if token is an Indonesian number word"""
        number_words = {'nol', 'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'delapan', 'sembilan', 'sepuluh', 'sebelas', 'puluh', 'seratus', 'seribu', 'juta', 'miliar'}
        return token in number_words

    def named_entity_recognition(self, text: str) -> List[Tuple]:
        """Enhanced Named Entity Recognition for Indonesian text"""
        entities = []
        
        # Person names patterns
        person_patterns = [
            r'\b(?:Bapak|Pak|Ibu|Bu|Dr\.?|Prof\.?|H\.?|Hj\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            r'\b([A-Z][a-z]+\s+(?:bin|binti)\s+[A-Z][a-z]+)\b',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b'
        ]
        
        for pattern in person_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    name = match[0] if match[0] else match
                else:
                    name = match
                if not self._is_common_word(name.lower()) and len(name.split()) <= 3:
                    entities.append((name.strip(), 'PERSON'))
        
        # Indonesian locations
        indonesian_locations = {
            'Jakarta', 'Surabaya', 'Bandung', 'Medan', 'Semarang', 'Makassar', 'Palembang',
            'Tangerang', 'Depok', 'Bekasi', 'Bogor', 'Batam', 'Pekanbaru', 'Malang', 'Padang',
            'Aceh', 'Sumatera Utara', 'Sumatera Barat', 'Riau', 'Jambi', 'Sumatera Selatan',
            'Bengkulu', 'Lampung', 'DKI Jakarta', 'Jawa Barat', 'Jawa Tengah', 'Jawa Timur',
            'Banten', 'Bali', 'Nusa Tenggara Barat', 'Nusa Tenggara Timur', 'Kalimantan Barat',
            'Kalimantan Tengah', 'Kalimantan Selatan', 'Kalimantan Timur', 'Sulawesi Utara',
            'Sulawesi Tengah', 'Sulawesi Selatan', 'Sulawesi Tenggara', 'Gorontalo', 'Maluku',
            'Maluku Utara', 'Papua Barat', 'Papua'
        }
        
        # Location detection
        location_pattern = r'\b(' + '|'.join(re.escape(loc) for loc in indonesian_locations) + r')\b'
        location_matches = re.finditer(location_pattern, text, re.IGNORECASE)
        for match in location_matches:
            entities.append((match.group(1), 'LOCATION'))
        
        # Organizations patterns
        org_patterns = [
            r'\b(PT\.?\s+[A-Z][a-zA-Z\s]+(?:Tbk\.?)?)\b',
            r'\b(CV\.?\s+[A-Z][a-zA-Z\s]+)\b',
            r'\b(Universitas\s+[A-Z][a-zA-Z\s]+)\b',
            r'\b(Institut\s+[A-Z][a-zA-Z\s]+)\b',
            r'\b(Kementerian\s+[A-Z][a-zA-Z\s]+)\b',
            r'\b(Bank\s+[A-Z][a-zA-Z\s]+)\b',
        ]
        
        for pattern in org_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                org_name = match.group(1).strip()
                if len(org_name) > 3:
                    entities.append((org_name, 'ORGANIZATION'))
        
        # Dates (Indonesian format)
        date_patterns = [
            r'\b(\d{1,2}\s+(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\s+\d{4})\b',
            r'\b(\d{1,2}/\d{1,2}/\d{4})\b',
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append((match.group(1), 'DATE'))
        
        # Money/Currency
        money_patterns = [
            r'\b(Rp\.?\s*\d+(?:\.\d{3})*(?:,\d+)?)\b',
            r'\b(\d+(?:\.\d{3})*(?:,\d+)?\s*rupiah)\b',
        ]
        
        for pattern in money_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append((match.group(1), 'MONEY'))
        
        # Remove duplicates
        seen = set()
        unique_entities = []
        for entity, label in entities:
            entity_key = (entity.lower(), label)
            if entity_key not in seen:
                seen.add(entity_key)
                unique_entities.append((entity, label))
        
        return unique_entities
    
    def _is_common_word(self, word: str) -> bool:
        """Check if word is a common Indonesian word"""
        common_words = {
            'yang', 'dan', 'di', 'ke', 'dari', 'dalam', 'untuk', 'pada', 'dengan', 'oleh',
            'adalah', 'akan', 'telah', 'sudah', 'dapat', 'bisa', 'juga', 'ini', 'itu',
            'pemerintah', 'indonesia', 'jakarta', 'presiden', 'menteri', 'bapak', 'ibu'
        }
        return word.lower() in common_words

    def build_ngram_model(self, documents: List[str], n: int = 2) -> Dict:
        """Build n-gram language model for next word prediction"""
        ngram_counts = defaultdict(lambda: defaultdict(int))
        ngram_totals = defaultdict(int)
        
        for doc in documents:
            words = doc.split()
            # Add sentence markers
            words = ['<START>'] * (n-1) + words + ['<END>']
            
            # Count n-grams
            for i in range(len(words) - n + 1):
                context = tuple(words[i:i+n-1])
                next_word = words[i+n-1]
                
                ngram_counts[context][next_word] += 1
                ngram_totals[context] += 1
        
        return {
            'ngram_counts': dict(ngram_counts),
            'ngram_totals': dict(ngram_totals),
            'n': n
        }
    
    def predict_next_word(self, model: Dict, context: List[str], top_k: int = 5) -> List[Tuple]:
        """Predict next word given context using n-gram model"""
        n = model['n']
        if len(context) >= n - 1:
            context = context[-(n-1):]
        else:
            context = ['<START>'] * (n - 1 - len(context)) + context
        
        context_tuple = tuple(context)
        
        if context_tuple not in model['ngram_counts']:
            return [("No prediction available", 0.0)]
        
        # Calculate probabilities
        candidates = model['ngram_counts'][context_tuple]
        total_count = model['ngram_totals'][context_tuple]
        
        # Sort by probability
        predictions = []
        for word, count in candidates.items():
            probability = count / total_count
            predictions.append((word, probability))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]

    def generate_visualizations(self, results: Dict):
        """Generate comprehensive visualizations and save them as files"""
        print("\n📊 GENERATING VISUALIZATIONS...")
        output_dir = 'indonesian_news_analysis_output'
        
        # Set style for better looking plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Word Frequency Bar Chart
        plt.figure(figsize=(12, 8))
        top_words = results['global_word_freq'].most_common(20)
        words, counts = zip(*top_words)
        
        bars = plt.bar(words, counts, color=plt.cm.viridis(np.linspace(0, 1, len(words))))
        plt.title('Top 20 Most Frequent Words in Indonesian News', fontsize=16, fontweight='bold')
        plt.xlabel('Words', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/word_frequency_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Word Cloud
        try:
            all_text = ' '.join([' '.join(article['stemmed']) for article in results['processed_articles']])
            
            wordcloud = WordCloud(
                width=1200, height=800, 
                background_color='white',
                max_words=100,
                colormap='viridis',
                relative_scaling=0.5,
                random_state=42
            ).generate(all_text)
            
            plt.figure(figsize=(15, 10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Indonesian News Word Cloud', fontsize=20, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/wordcloud.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not generate word cloud: {e}")
        
        # 3. TF-IDF Heatmap
        plt.figure(figsize=(14, 10))
        
        # Get top words and documents for heatmap
        top_global_words = [word for word, _ in results['global_word_freq'].most_common(15)]
        tf_idf_matrix = results['tf_idf_matrix']
        vocabulary = results['vocabulary']
        
        # Create subset matrix
        word_indices = [vocabulary.index(word) for word in top_global_words if word in vocabulary]
        doc_indices = list(range(min(10, tf_idf_matrix.shape[0])))
        
        if word_indices and doc_indices:
            subset_matrix = tf_idf_matrix[np.ix_(doc_indices, word_indices)]
            
            sns.heatmap(
                subset_matrix.T, 
                xticklabels=[f"Doc {i+1}" for i in doc_indices],
                yticklabels=[top_global_words[i] for i, _ in enumerate(word_indices)],
                cmap='YlOrRd',
                annot=True,
                fmt='.3f',
                cbar_kws={'label': 'TF-IDF Score'}
            )
            
            plt.title('TF-IDF Heatmap: Top Words across Documents', fontsize=16, fontweight='bold')
            plt.xlabel('Documents', fontsize=12)
            plt.ylabel('Words', fontsize=12)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/tfidf_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Document Length Distribution
        doc_lengths = [len(article['tokens']) for article in results['processed_articles']]
        
        plt.figure(figsize=(10, 6))
        plt.hist(doc_lengths, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of Document Lengths', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Tokens', fontsize=12)
        plt.ylabel('Number of Documents', fontsize=12)
        plt.axvline(np.mean(doc_lengths), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(doc_lengths):.1f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/document_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. POS Tag Distribution
        all_pos_tags = []
        for article in results['processed_articles']:
            pos_tags = self.pos_tagging_indonesian(article['cleaned'])
            all_pos_tags.extend([tag for _, tag in pos_tags])
        
        pos_counter = Counter(all_pos_tags)
        
        plt.figure(figsize=(12, 8))
        pos_labels, pos_counts = zip(*pos_counter.most_common(10))
        
        bars = plt.bar(pos_labels, pos_counts, color=plt.cm.Set3(np.linspace(0, 1, len(pos_labels))))
        plt.title('Part-of-Speech Tag Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('POS Tags', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, count in zip(bars, pos_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pos_tag_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Named Entity Distribution
        all_entities = []
        for article in results['processed_articles']:
            entities = self.named_entity_recognition(article['original'])
            all_entities.extend([label for _, label in entities])
        
        if all_entities:
            entity_counter = Counter(all_entities)
            
            plt.figure(figsize=(10, 6))
            entity_labels, entity_counts = zip(*entity_counter.most_common())
            
            colors = plt.cm.Set2(np.linspace(0, 1, len(entity_labels)))
            bars = plt.bar(entity_labels, entity_counts, color=colors)
            
            plt.title('Named Entity Distribution', fontsize=16, fontweight='bold')
            plt.xlabel('Entity Types', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.xticks(rotation=45)
            
            # Add value labels
            for bar, count in zip(bars, entity_counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/named_entity_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 7. N-gram Frequency Chart
        ngram_model = results['ngram_model']
        
        # Get most common bigrams
        bigram_freq = Counter()
        for context, next_words in ngram_model['ngram_counts'].items():
            if context != ('<START>',):  # Skip sentence start
                for word, count in next_words.items():
                    if word != '<END>':  # Skip sentence end
                        bigram = ' '.join(context) + ' ' + word
                        bigram_freq[bigram] += count
        
        plt.figure(figsize=(12, 8))
        top_bigrams = bigram_freq.most_common(15)
        if top_bigrams:
            bigrams, counts = zip(*top_bigrams)
            
            bars = plt.barh(bigrams, counts, color=plt.cm.plasma(np.linspace(0, 1, len(bigrams))))
            plt.title('Top 15 Most Frequent Bigrams', fontsize=16, fontweight='bold')
            plt.xlabel('Frequency', fontsize=12)
            plt.ylabel('Bigrams', fontsize=12)
            
            # Add value labels
            for bar, count in zip(bars, counts):
                plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                        str(count), ha='left', va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/bigram_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 8. Interactive Plotly Visualization - Document Similarity
        try:
            # Perform PCA on TF-IDF matrix
            pca = PCA(n_components=2, random_state=42)
            tfidf_2d = pca.fit_transform(results['tf_idf_matrix'])
            
            # Create interactive scatter plot
            fig = go.Figure()
            
            colors = px.colors.qualitative.Set3
            
            for i, article in enumerate(results['processed_articles']):
                fig.add_trace(go.Scatter(
                    x=[tfidf_2d[i, 0]],
                    y=[tfidf_2d[i, 1]],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=colors[i % len(colors)],
                        line=dict(width=1, color='black')
                    ),
                    text=article['title'][:50] + '...',
                    hovertemplate='<b>%{text}</b><br>' +
                                'PC1: %{x:.3f}<br>' +
                                'PC2: %{y:.3f}<br>' +
                                'Source: ' + article['source'] +
                                '<extra></extra>',
                    name=f'Doc {i+1}'
                ))
            
            fig.update_layout(
                title='Document Similarity Visualization (PCA of TF-IDF)',
                xaxis_title=f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)',
                yaxis_title=f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)',
                font=dict(size=12),
                showlegend=False,
                width=800,
                height=600
            )
            
            fig.write_html(f'{output_dir}/interactive_document_similarity.html')
            
        except Exception as e:
            print(f"Warning: Could not generate PCA visualization: {e}")
        
        print("✅ All visualizations generated successfully!")

    def export_data_to_files(self, results: Dict):
        """Export analysis data to various file formats"""
        print("\n💾 EXPORTING DATA TO FILES...")
        output_dir = 'indonesian_news_analysis_output'
        
        # 1. Export Articles to Excel
        articles_data = []
        for i, article in enumerate(results['articles']):
            processed = results['processed_articles'][i]
            top_words = results['top_words_per_doc'][i]
            
            articles_data.append({
                'Article_ID': i + 1,
                'Title': article['title'],
                'Source': article['source'],
                'URL': article.get('url', ''),
                'Published_Date': article.get('published_date', ''),
                'Original_Length': len(article['content']),
                'Cleaned_Length': len(processed['cleaned']),
                'Token_Count': len(processed['tokens']),
                'Unique_Token_Count': len(set(processed['tokens'])),
                'After_Stopwords': len(processed['no_stopwords']),
                'After_Stemming': len(processed['stemmed']),
                'Top_5_Keywords': ', '.join([word for word, _ in top_words[:5]]),
                'Content_Preview': article['content'][:200] + '...' if len(article['content']) > 200 else article['content']
            })
        
        df_articles = pd.DataFrame(articles_data)
        
        # 2. Export Word Frequency Data
        word_freq_data = []
        for word, freq in results['global_word_freq'].most_common(100):
            word_freq_data.append({
                'Word': word,
                'Frequency': freq,
                'Percentage': (freq / results['total_words']) * 100
            })
        
        df_word_freq = pd.DataFrame(word_freq_data)
        
        # 3. Export POS Tags Data
        pos_data = []
        for i, article in enumerate(results['processed_articles']):
            pos_tags = self.pos_tagging_indonesian(article['cleaned'])
            pos_counter = Counter([tag for _, tag in pos_tags])
            
            for pos_tag, count in pos_counter.items():
                pos_data.append({
                    'Article_ID': i + 1,
                    'POS_Tag': pos_tag,
                    'Count': count,
                    'Percentage': (count / len(pos_tags)) * 100 if pos_tags else 0
                })
        
        df_pos = pd.DataFrame(pos_data)
        
        # 4. Export Named Entities Data
        entity_data = []
        for i, article in enumerate(results['processed_articles']):
            entities = self.named_entity_recognition(article['original'])
            
            for entity, label in entities:
                entity_data.append({
                    'Article_ID': i + 1,
                    'Entity': entity,
                    'Entity_Type': label
                })
        
        df_entities = pd.DataFrame(entity_data)
        
        # 5. Export N-gram Data
        ngram_data = []
        ngram_model = results['ngram_model']
        
        for context, next_words in ngram_model['ngram_counts'].items():
            context_str = ' '.join(context)
            total_count = ngram_model['ngram_totals'][context]
            
            for word, count in next_words.items():
                probability = count / total_count
                ngram_data.append({
                    'Context': context_str,
                    'Next_Word': word,
                    'Count': count,
                    'Probability': probability
                })
        
        df_ngram = pd.DataFrame(ngram_data).sort_values('Count', ascending=False)
        
        # Save all data to Excel with multiple sheets
        with pd.ExcelWriter(f'{output_dir}/indonesian_news_analysis_data.xlsx', engine='openpyxl') as writer:
            df_articles.to_excel(writer, sheet_name='Articles_Summary', index=False)
            df_word_freq.to_excel(writer, sheet_name='Word_Frequency', index=False)
            df_pos.to_excel(writer, sheet_name='POS_Tags', index=False)
            df_entities.to_excel(writer, sheet_name='Named_Entities', index=False)
            df_ngram.head(1000).to_excel(writer, sheet_name='Ngram_Data', index=False)  # Limit to top 1000
        
        # 6. Export preprocessing examples to CSV
        preprocessing_examples = []
        for i in range(min(5, len(results['processed_articles']))):
            article = results['processed_articles'][i]
            preprocessing_examples.append({
                'Article_ID': i + 1,
                'Original_Sample': article['original'][:300] + '...',
                'Cleaned_Sample': article['cleaned'][:300] + '...',
                'Tokens_Sample': str(article['tokens'][:20]) + '...',
                'No_Stopwords_Sample': str(article['no_stopwords'][:20]) + '...',
                'Stemmed_Sample': str(article['stemmed'][:20]) + '...'
            })
        
        df_preprocessing = pd.DataFrame(preprocessing_examples)
        df_preprocessing.to_csv(f'{output_dir}/preprocessing_examples.csv', index=False, encoding='utf-8-sig')
        
        # 7. Export summary statistics to JSON
        statistics = {
            'analysis_date': datetime.now().isoformat(),
            'total_articles': len(results['articles']),
            'total_words': results['total_words'],
            'unique_words': results['total_unique_words'],
            'average_words_per_document': results['total_words'] / len(results['articles']),
            'vocabulary_size': len(results['vocabulary']),
            'top_10_words': [{'word': word, 'frequency': freq} 
                           for word, freq in results['global_word_freq'].most_common(10)],
            'articles_by_source': dict(Counter([article['source'] for article in results['articles']])),
            'preprocessing_statistics': {
                'average_tokens_before_cleaning': np.mean([len(article['tokens']) for article in results['processed_articles']]),
                'average_tokens_after_stopwords': np.mean([len(article['no_stopwords']) for article in results['processed_articles']]),
                'average_tokens_after_stemming': np.mean([len(article['stemmed']) for article in results['processed_articles']]),
            }
        }
        
        with open(f'{output_dir}/analysis_statistics.json', 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False, default=str)
        
        print("✅ All data exported successfully!")

    def generate_comprehensive_report(self, results: Dict):
        """Generate a comprehensive HTML report with embedded visualizations"""
        print("\n📋 GENERATING COMPREHENSIVE HTML REPORT...")
        output_dir = 'indonesian_news_analysis_output'
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="id">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Laporan Analisis Teks Berita Indonesia</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    border-left: 4px solid #3498db;
                    padding-left: 15px;
                    margin-top: 30px;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .stat-box {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                }}
                .stat-number {{
                    font-size: 2.5em;
                    font-weight: bold;
                    display: block;
                }}
                .stat-label {{
                    font-size: 0.9em;
                    opacity: 0.9;
                }}
                .visualization {{
                    text-align: center;
                    margin: 30px 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                    border-radius: 8px;
                }}
                .visualization img {{
                    max-width: 100%;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .article-summary {{
                    background-color: #ecf0f1;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                    border-left: 4px solid #3498db;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 50px;
                    padding: 20px;
                    background-color: #2c3e50;
                    color: white;
                    border-radius: 8px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🇮🇩 LAPORAN ANALISIS TEKS BERITA INDONESIA</h1>
                
                <div class="stats-grid">
                    <div class="stat-box">
                        <span class="stat-number">{len(results['articles'])}</span>
                        <span class="stat-label">Total Artikel</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-number">{results['total_words']:,}</span>
                        <span class="stat-label">Total Kata</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-number">{results['total_unique_words']:,}</span>
                        <span class="stat-label">Kata Unik</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-number">{results['total_words']/len(results['articles']):.0f}</span>
                        <span class="stat-label">Rata-rata Kata/Artikel</span>
                    </div>
                </div>
                
                <h2>📊 Statistik Kata Paling Sering</h2>
                <table>
                    <tr><th>Peringkat</th><th>Kata</th><th>Frekuensi</th><th>Persentase</th></tr>
        """
        
        # Add top words table
        for i, (word, freq) in enumerate(results['global_word_freq'].most_common(15), 1):
            percentage = (freq / results['total_words']) * 100
            html_content += f"<tr><td>{i}</td><td>{word}</td><td>{freq}</td><td>{percentage:.2f}%</td></tr>"
        
        html_content += """
                </table>
                
                <h2>📈 Visualisasi Data</h2>
        """
        
        # Add visualizations
        visualizations = [
            ('word_frequency_chart.png', 'Grafik Frekuensi Kata Paling Sering'),
            ('wordcloud.png', 'Word Cloud Berita Indonesia'),
            ('tfidf_heatmap.png', 'Heatmap TF-IDF'),
            ('document_length_distribution.png', 'Distribusi Panjang Dokumen'),
            ('pos_tag_distribution.png', 'Distribusi POS Tags'),
            ('named_entity_distribution.png', 'Distribusi Named Entities'),
            ('bigram_frequency.png', 'Frekuensi Bigram Paling Sering')
        ]
        
        for img_file, caption in visualizations:
            if os.path.exists(f'{output_dir}/{img_file}'):
                html_content += f"""
                <div class="visualization">
                    <h3>{caption}</h3>
                    <img src="{img_file}" alt="{caption}">
                </div>
                """
        
        html_content += """
                <h2>📰 Ringkasan Artikel</h2>
        """
        
        # Add article summaries
        for i, article in enumerate(results['processed_articles'][:10]):  # Show first 10 articles
            top_words = ', '.join([word for word, _ in results['top_words_per_doc'][i][:5]])
            entities = self.named_entity_recognition(article['original'])
            entity_str = ', '.join([entity for entity, _ in entities[:3]])
            
            html_content += f"""
                <div class="article-summary">
                    <h4>Artikel {i+1}: {article['title']}</h4>
                    <p><strong>Sumber:</strong> {article['source']}</p>
                    <p><strong>Kata Kunci:</strong> {top_words}</p>
                    <p><strong>Entitas:</strong> {entity_str if entity_str else 'Tidak ditemukan'}</p>
                    <p><strong>Jumlah Kata:</strong> {len(article['tokens'])} → {len(article['stemmed'])} (setelah preprocessing)</p>
                </div>
            """
        
        html_content += f"""
                <h2>🔮 Prediksi Kata Berikutnya (N-gram)</h2>
                <p>Contoh prediksi kata berikutnya menggunakan model bigram:</p>
                <table>
                    <tr><th>Konteks</th><th>Prediksi Kata</th><th>Probabilitas</th></tr>
        """
        
        # Add n-gram predictions
        test_contexts = [['pemerintah'], ['teknologi'], ['program'], ['indonesia'], ['sistem']]
        for context in test_contexts:
            predictions = self.predict_next_word(results['ngram_model'], context, top_k=1)
            if predictions and predictions[0][1] > 0:
                word, prob = predictions[0]
                html_content += f"<tr><td>{' '.join(context)}</td><td>{word}</td><td>{prob:.3f}</td></tr>"
        
        html_content += f"""
                </table>
                
                <h2>📄 File Data yang Dihasilkan</h2>
                <ul>
                    <li><strong>indonesian_news_analysis_data.xlsx</strong> - Data lengkap dalam format Excel</li>
                    <li><strong>preprocessing_examples.csv</strong> - Contoh tahapan preprocessing</li>
                    <li><strong>analysis_statistics.json</strong> - Statistik analisis dalam format JSON</li>
                    <li><strong>interactive_document_similarity.html</strong> - Visualisasi interaktif similaritas dokumen</li>
                    <li>Berbagai file gambar visualisasi (.png)</li>
                </ul>
                
                <div class="footer">
                    <p>Laporan dibuat pada: {datetime.now().strftime('%d %B %Y, %H:%M:%S')}</p>
                    <p>Sistem Analisis Teks Berita Indonesia - RKK305 Pemrosesan Bahasa Alami</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(f'{output_dir}/comprehensive_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("✅ Comprehensive HTML report generated successfully!")

    def analyze_documents(self, num_articles: int = 30):
        """Complete analysis pipeline with comprehensive reporting"""
        print("="*60)
        print("🇮🇩 SISTEM ANALISIS TEKS BERITA INDONESIA 🇮🇩")
        print("="*60)
        
        # 1. Scrape articles
        print("\n1️⃣ MENGAMBIL ARTIKEL BERITA...")
        self.articles = self.scrape_indonesian_news(num_articles)
        print(f"✅ Berhasil mengambil {len(self.articles)} artikel")
        
        # 2. Preprocess articles
        print("\n2️⃣ PREPROCESSING TEKS...")
        self.processed_articles = []
        
        for i, article in enumerate(self.articles):
            processed = self.preprocess_text(article['content'])
            processed['title'] = article['title']
            processed['source'] = article['source']
            processed['url'] = article.get('url', '')
            self.processed_articles.append(processed)
            print(f"   Memproses artikel {i+1}/{len(self.articles)}")
        
        print("✅ Preprocessing selesai")
        
        # 3. Calculate TF-IDF
        print("\n3️⃣ MENGHITUNG TF-IDF...")
        stemmed_docs = [' '.join(article['stemmed']) for article in self.processed_articles]
        tf_idf_matrix, vocabulary, tf_matrix, idf_array = self.calculate_tf_idf(stemmed_docs)
        
        # Get top words per document
        top_words_per_doc = self.get_top_words_per_document(tf_idf_matrix, vocabulary, top_k=10)
        
        print("✅ TF-IDF calculation selesai")
        
        # 4. Build N-gram model
        print("\n4️⃣ MEMBANGUN MODEL N-GRAM...")
        ngram_model = self.build_ngram_model(stemmed_docs, n=2)  # Bigram model
        print("✅ Model N-gram selesai dibuat")
        
        # Calculate additional statistics
        total_words = sum(len(article['tokens']) for article in self.processed_articles)
        total_unique_words = len(vocabulary)
        global_word_freq = Counter()
        for article in self.processed_articles:
            global_word_freq.update(article['stemmed'])
        
        # Compile results
        results = {
            'articles': self.articles,
            'processed_articles': self.processed_articles,
            'tf_idf_matrix': tf_idf_matrix,
            'vocabulary': vocabulary,
            'ngram_model': ngram_model,
            'top_words_per_doc': top_words_per_doc,
            'global_word_freq': global_word_freq,
            'total_words': total_words,
            'total_unique_words': total_unique_words
        }
        
        # 5. Generate visualizations
        self.generate_visualizations(results)
        
        # 6. Export data to files
        self.export_data_to_files(results)
        
        # 7. Generate comprehensive HTML report
        self.generate_comprehensive_report(results)
        
        # 8. Display sample results in console
        print("\n5️⃣ CONTOH HASIL ANALISIS")
        print("-" * 50)
        
        for i in range(min(3, len(self.processed_articles))):
            article = self.processed_articles[i]
            print(f"\n📰 ARTIKEL {i+1}: {article['title'][:50]}...")
            print(f"📍 Sumber: {article['source']}")
            
            # Show preprocessing stages
            print(f"🔤 Original text (100 char): {article['original'][:100]}...")
            print(f"🧹 Cleaned text (100 char): {article['cleaned'][:100]}...")
            print(f"📝 Tokens (first 10): {article['tokens'][:10]}")
            print(f"🚫 After stopword removal: {article['no_stopwords'][:10]}")
            print(f"🌱 After stemming: {article['stemmed'][:10]}")
            
            # Show top TF-IDF words
            print(f"📊 Top TF-IDF words:")
            for word, score in top_words_per_doc[i][:5]:
                print(f"   • {word}: {score:.4f}")
            
            # POS Tagging
            print(f"🏷️ POS Tags (first 10):")
            pos_tags = self.pos_tagging_indonesian(article['cleaned'])
            for word, tag in pos_tags[:10]:
                print(f"   • {word}: {tag}")
            
            # Named Entity Recognition
            print(f"🏢 Named Entities:")
            entities = self.named_entity_recognition(article['original'])
            for entity, label in entities[:5]:
                print(f"   • {entity}: {label}")
            
            print("-" * 50)
        
        # 9. Test word prediction
        print("\n6️⃣ CONTOH PREDIKSI KATA BERIKUTNYA")
        test_contexts = [
            ['pemerintah'],
            ['teknologi'],
            ['program'],
            ['indonesia'],
            ['sistem']
        ]
        
        for context in test_contexts:
            predictions = self.predict_next_word(ngram_model, context, top_k=3)
            print(f"\n🔮 Context: {' '.join(context)}")
            print("   Prediksi kata berikutnya:")
            for word, prob in predictions:
                print(f"   • {word}: {prob:.3f}")
        
        # 10. Summary statistics
        print("\n7️⃣ STATISTIK RINGKASAN")
        print("-" * 40)
        
        avg_words_per_doc = total_words / len(self.processed_articles)
        
        print(f"📊 Total dokumen: {len(self.processed_articles)}")
        print(f"📊 Total kata: {total_words:,}")
        print(f"📊 Kata unik: {total_unique_words:,}")
        print(f"📊 Rata-rata kata per dokumen: {avg_words_per_doc:.1f}")
        
        # Top words globally
        print(f"\n🔤 10 Kata Paling Sering Muncul:")
        for word, freq in global_word_freq.most_common(10):
            print(f"   • {word}: {freq}")
        
        print("\n" + "="*60)
        print("✅ ANALISIS SELESAI!")
        print("📁 Semua file hasil analisis tersimpan di folder: 'indonesian_news_analysis_output/'")
        print("📊 Buka file 'comprehensive_report.html' untuk melihat laporan lengkap")
        print("📈 Buka 'interactive_document_similarity.html' untuk visualisasi interaktif")
        print("="*60)
        
        return results

# Main execution
def main():
    """Main function to run the Indonesian News Analysis System"""
    try:
        # Create analyzer instance
        analyzer = IndonesianNewsAnalyzer()
        
        # Run complete analysis with comprehensive reporting
        results = analyzer.analyze_documents(num_articles=30)
        
        # Print final summary of generated files
        output_dir = 'indonesian_news_analysis_output'
        print(f"\n📄 RINGKASAN FILE YANG DIHASILKAN:")
        print("=" * 50)
        
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            files.sort()
            
            file_categories = {
                'Data Files': ['.xlsx', '.csv', '.json'],
                'Visualizations': ['.png', '.jpg'],
                'Reports': ['.html']
            }
            
            for category, extensions in file_categories.items():
                category_files = [f for f in files if any(f.endswith(ext) for ext in extensions)]
                if category_files:
                    print(f"\n📂 {category}:")
                    for file in category_files:
                        file_path = os.path.join(output_dir, file)
                        size = os.path.getsize(file_path) / 1024  # Size in KB
                        print(f"   • {file} ({size:.1f} KB)")
            
            total_size = sum(os.path.getsize(os.path.join(output_dir, f)) 
                           for f in files) / (1024 * 1024)  # Total size in MB
            print(f"\n💾 Total ukuran file: {total_size:.2f} MB")
        
        print(f"\n🎉 ANALISIS LENGKAP BERHASIL DISELESAIKAN!")
        print(f"📁 Semua file tersimpan di folder: '{output_dir}/'")
        print(f"🌐 Buka 'comprehensive_report.html' untuk laporan interaktif")
        print(f"📈 Buka 'interactive_document_similarity.html' untuk visualisasi interaktif")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to provide helpful error messages
        if "No module named" in str(e):
            print("\n💡 SOLUSI: Install paket yang dibutuhkan:")
            print("pip install nltk spacy sastrawi scikit-learn matplotlib seaborn pandas numpy")
            print("pip install requests beautifulsoup4 feedparser wordcloud plotly openpyxl")
            print("python -m spacy download en_core_web_sm")
        elif "Permission denied" in str(e):
            print("\n💡 SOLUSI: Pastikan Anda memiliki izin menulis di direktori ini")
        elif "Connection" in str(e) or "timeout" in str(e).lower():
            print("\n💡 SOLUSI: Periksa koneksi internet untuk mengambil berita online")
        
        return False

if __name__ == "__main__":
    main()