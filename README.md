# AI-Powered Text Summarizer

This project is an AI-powered text summarizer that can generate concise summaries of news articles or any lengthy text. It uses natural language processing techniques to identify and extract the most important sentences from the input text.

## Features

- Text summarization using TextRank algorithm
- Customizable summary length
- Support for both plain text and URL inputs
- Web interface for easy interaction

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download required NLTK data:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

## Usage

1. Start the web server:
   ```bash
   python app.py
   ```
2. Open your browser and navigate to `http://localhost:5000`
3. Paste your text or enter a URL and choose your desired summary length
4. Click "Summarize" to generate the summary

## How it Works

The summarizer uses the TextRank algorithm, which is inspired by Google's PageRank. It:
1. Breaks the text into sentences
2. Creates a graph where sentences are nodes
3. Calculates similarity between sentences
4. Ranks sentences based on their importance
5. Extracts top-ranked sentences to create the summary

## Technologies Used

- Python 3.8+
- NLTK for natural language processing
- NetworkX for graph-based algorithms
- Flask for web interface
- scikit-learn for text processing 