from flask import Flask, render_template, request, jsonify
from summarizer import TextSummarizer
import os

app = Flask(__name__)
summarizer = TextSummarizer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        num_sentences = data.get('num_sentences')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        if num_sentences:
            try:
                num_sentences = int(num_sentences)
                if num_sentences < 1:
                    raise ValueError
            except ValueError:
                return jsonify({'error': 'Invalid number of sentences'}), 400
                
        summary = summarizer.summarize(text, num_sentences=num_sentences)
        return jsonify({'summary': summary})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True) 