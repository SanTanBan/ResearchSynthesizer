from flask import Flask, request, jsonify
from flask_cors import CORS
from paper_processor import PaperProcessor
from api_client import ArxivAPIClient
from keyword_extractor import KeywordExtractor
from cache_manager import CacheManager
from abstract_filter import AbstractFilter
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
abstract_filter = AbstractFilter()

@app.route('/api/research', methods=['GET'])
def get_research_papers():
    try:
        # Get parameters from request
        research_question = request.args.get('question', '')
        criteria = request.args.get('criteria', '')

        if not research_question:
            return jsonify({'error': 'Research question is required'}), 400

        # Step 1: Process the initial query
        logging.info("Processing initial paper search")
        results = paper_processor.process_query(research_question, criteria)
        initial_count = len(results['papers'])
        logging.info(f"Found {initial_count} papers in initial search")

        # Step 2: Apply abstract filter
        logging.info("Applying abstract relevance filter")
        filtered_papers = abstract_filter.filter_papers(results['papers'], research_question)
        filtered_count = len(filtered_papers)
        logging.info(f"Kept {filtered_count} papers after filtering")

        # Update results with filtered papers
        results['papers'] = filtered_papers
        results['total_results'] = filtered_count
        results['initial_results'] = initial_count

        # Return JSON response
        return jsonify({
            'status': 'success',
            'keywords': results['keywords'],
            'total_results': results['total_results'],
            'initial_results': results['initial_results'],
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