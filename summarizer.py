import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import networkx as nx
from typing import List, Optional, Union, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from bs4 import BeautifulSoup
import html2text
import validators
import re
from functools import lru_cache
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextSummarizer:
    def __init__(self):
        """Initialize the TextSummarizer with NLTK downloads and setup."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = True
        self.h2t.ignore_images = True
        
    def _fetch_url_content(self, url: str) -> str:
        """
        Fetch and extract main content from a URL.
        
        Args:
            url (str): URL to fetch content from
            
        Returns:
            str: Extracted text content
            
        Raises:
            ValueError: If URL is invalid or content cannot be fetched
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Convert HTML to markdown and then to plain text
            text = self.h2t.handle(str(soup))
            return self._clean_text(text)
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {str(e)}")
            raise ValueError(f"Could not fetch content from URL: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        # Replace multiple newlines and spaces
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove special characters but keep sentence-ending punctuation
        text = re.sub(r'[^\w\s.!?]', '', text)
        
        # Normalize sentence endings
        text = re.sub(r'([.!?])\s*', r'\1 ', text)
        
        return text.strip()

    @lru_cache(maxsize=100)
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess the text by splitting it into sentences and cleaning them.
        
        Args:
            text (str): Input text to be preprocessed
            
        Returns:
            List[str]: List of preprocessed sentences
        """
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        # Clean and normalize sentences
        clean_sentences = []
        for sentence in sentences:
            # Convert to lowercase and split into words
            words = word_tokenize(sentence.lower())
            # Remove stopwords and non-alphabetic characters
            words = [word for word in words if word.isalnum() and word not in self.stop_words]
            if words:  # Only add non-empty sentences
                clean_sentences.append(' '.join(words))
            
        return clean_sentences

    def build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """
        Build a similarity matrix between sentences using TF-IDF and cosine similarity.
        
        Args:
            sentences (List[str]): List of preprocessed sentences
            
        Returns:
            np.ndarray: Similarity matrix
        """
        if not sentences:
            return np.array([[]])
            
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(min_df=1)
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            logger.warning("Could not vectorize sentences, returning empty matrix")
            return np.zeros((len(sentences), len(sentences)))
        
        # Calculate cosine similarity matrix
        similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
        
        # Normalize similarity scores
        np.fill_diagonal(similarity_matrix, 0)
        row_sums = similarity_matrix.sum(axis=1)
        non_zero_rows = row_sums != 0
        similarity_matrix[non_zero_rows] = similarity_matrix[non_zero_rows] / row_sums[non_zero_rows, np.newaxis]
        
        return similarity_matrix

    def rank_sentences(self, similarity_matrix: np.ndarray) -> List[float]:
        """
        Rank sentences using the PageRank algorithm.
        
        Args:
            similarity_matrix (np.ndarray): Similarity matrix of sentences
            
        Returns:
            List[float]: List of sentence scores
        """
        if similarity_matrix.size == 0:
            return []
            
        # Create graph from similarity matrix
        nx_graph = nx.from_numpy_array(similarity_matrix)
        
        try:
            # Calculate PageRank with error handling
            scores = nx.pagerank(nx_graph, max_iter=200)
            return list(scores.values())
        except nx.PowerIterationFailedConvergence:
            logger.warning("PageRank failed to converge, using degree centrality instead")
            scores = nx.degree_centrality(nx_graph)
            return list(scores.values())

    def summarize(self, input_text: str, num_sentences: Optional[int] = None, 
                 ratio: float = 0.3) -> Dict[str, Union[str, int]]:
        """
        Generate a summary of the input text or URL.
        
        Args:
            input_text (str): Input text or URL to summarize
            num_sentences (int, optional): Number of sentences in the summary
            ratio (float): Ratio of sentences to keep if num_sentences is not specified
            
        Returns:
            Dict[str, Union[str, int]]: Dictionary containing summary and metadata
        """
        # Check if input is URL
        if validators.url(input_text):
            try:
                text = self._fetch_url_content(input_text)
            except ValueError as e:
                return {"error": str(e)}
        else:
            text = self._clean_text(input_text)
        
        # Get original sentences for output
        original_sentences = sent_tokenize(text)
        total_sentences = len(original_sentences)
        
        if total_sentences <= 1:
            return {
                "summary": text,
                "original_length": len(text),
                "summary_length": len(text),
                "num_original_sentences": 1,
                "num_summary_sentences": 1
            }
            
        # Preprocess text
        clean_sentences = self.preprocess_text(text)
        
        # Build similarity matrix
        similarity_matrix = self.build_similarity_matrix(clean_sentences)
        
        # Rank sentences
        sentence_scores = self.rank_sentences(similarity_matrix)
        
        # Determine number of sentences for summary
        if num_sentences is None:
            num_sentences = max(1, int(total_sentences * ratio))
        else:
            num_sentences = min(num_sentences, total_sentences)
        
        # Get indices of top sentences
        ranked_sentences = [
            (score, idx) for idx, score in enumerate(sentence_scores)
        ]
        ranked_sentences.sort(reverse=True)
        
        # Select top sentences and sort them by original position
        top_sentences = sorted(
            [(idx, original_sentences[idx]) 
             for score, idx in ranked_sentences[:num_sentences]],
            key=lambda x: x[0]
        )
        
        # Combine sentences into summary
        summary = ' '.join(sentence for _, sentence in top_sentences)
        
        return {
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "num_original_sentences": total_sentences,
            "num_summary_sentences": num_sentences,
            "compression_ratio": round(len(summary) / len(text) * 100, 2)
        } 