from flask import Flask, render_template, request, jsonify
from summarizer import TextSummarizer
import os
import logging
from typing import Dict, Any
import validators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
summarizer = TextSummarizer()

def validate_input(data: Dict[str, Any]) -> tuple[str, int]:
    """
    Validate the input data for summarization.
    
    Args:
        data (Dict[str, Any]): Input data containing text and parameters
        
    Returns:
        tuple[str, int]: Error message and HTTP status code if validation fails
    """
    if not data:
        return "No data provided", 400
        
    text = data.get('text', '').strip()
    if not text:
        return "No text provided", 400
        
    # Validate URL if provided
    if validators.url(text):
        if not text.startswith(('http://', 'https://')):
            return "Invalid URL format", 400
    
    # Validate number of sentences
    num_sentences = data.get('num_sentences')
    if num_sentences:
        try:
            num_sentences = int(num_sentences)
            if num_sentences < 1:
                return "Number of sentences must be positive", 400
        except ValueError:
            return "Invalid number of sentences", 400
            
    return "", 200

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize_text():
    """Handle text summarization requests."""
    try:
        data = request.get_json()
        
        # Validate input
        error, status_code = validate_input(data)
        if error:
            return jsonify({'error': error}), status_code
            
        # Get parameters
        text = data['text'].strip()
        num_sentences = data.get('num_sentences')
        if num_sentences:
            num_sentences = int(num_sentences)
            
        # Generate summary
        result = summarizer.summarize(text, num_sentences=num_sentences)
        
        # Check for errors in summarization
        if 'error' in result:
            return jsonify({'error': result['error']}), 400
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in summarize_text: {str(e)}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}", exc_info=True)
    return jsonify({'error': 'An internal server error occurred'}), 500

if __name__ == '__main__':
    # Ensure required directories exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Add file handler for logging
    file_handler = logging.FileHandler('logs/app.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    app.logger.addHandler(file_handler)
    
    app.run(debug=True) 