from flask import Flask, request, jsonify
from flask_cors import CORS
from paper_processor import PaperProcessor
from api_client import ArxivAPIClient
from keyword_extractor import KeywordExtractor
from cache_manager import CacheManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize components
cache_manager = CacheManager()
keyword_extractor = KeywordExtractor()
arxiv_client = ArxivAPIClient()
paper_processor = PaperProcessor(keyword_extractor, arxiv_client, cache_manager)

@app.route('/api/research', methods=['GET'])
def get_research_papers():
    try:
        # Get parameters from request
        research_question = request.args.get('question', '')
        criteria = request.args.get('criteria', '')

        if not research_question:
            return jsonify({'error': 'Research question is required'}), 400

        # Process the query
        results = paper_processor.process_query(research_question, criteria)

        # Return JSON response
        return jsonify({
            'status': 'success',
            'keywords': results['keywords'],
            'total_results': results['total_results'],
            'papers': results['papers']
        })

    except Exception as e:
        logging.error(f"API Error: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)