import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import networkx as nx
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

class TextSummarizer:
    def __init__(self):
        """Initialize the TextSummarizer with NLTK downloads."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))

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
            # Reconstruct the sentence
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
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate cosine similarity matrix
        similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
        
        # Normalize similarity scores
        np.fill_diagonal(similarity_matrix, 0)
        row_sums = similarity_matrix.sum(axis=1)
        similarity_matrix = np.divide(similarity_matrix, row_sums[:, np.newaxis], 
                                    where=row_sums[:, np.newaxis] != 0)
        
        return similarity_matrix

    def rank_sentences(self, similarity_matrix: np.ndarray) -> List[float]:
        """
        Rank sentences using the PageRank algorithm.
        
        Args:
            similarity_matrix (np.ndarray): Similarity matrix of sentences
            
        Returns:
            List[float]: List of sentence scores
        """
        # Create graph from similarity matrix
        nx_graph = nx.from_numpy_array(similarity_matrix)
        
        # Calculate PageRank
        scores = nx.pagerank(nx_graph)
        
        return list(scores.values())

    def summarize(self, text: str, num_sentences: Optional[int] = None, 
                 ratio: float = 0.3) -> str:
        """
        Generate a summary of the input text.
        
        Args:
            text (str): Input text to summarize
            num_sentences (int, optional): Number of sentences in the summary
            ratio (float): Ratio of sentences to keep if num_sentences is not specified
            
        Returns:
            str: Generated summary
        """
        # Get original sentences for output
        original_sentences = sent_tokenize(text)
        
        if len(original_sentences) <= 1:
            return text
            
        # Preprocess text
        clean_sentences = self.preprocess_text(text)
        
        # Build similarity matrix
        similarity_matrix = self.build_similarity_matrix(clean_sentences)
        
        # Rank sentences
        sentence_scores = self.rank_sentences(similarity_matrix)
        
        # Determine number of sentences for summary
        if num_sentences is None:
            num_sentences = max(1, int(len(original_sentences) * ratio))
        else:
            num_sentences = min(num_sentences, len(original_sentences))
        
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
        
        return summary 