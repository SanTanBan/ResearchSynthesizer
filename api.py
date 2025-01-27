from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
from paper_processor import PaperProcessor
from api_client import ArxivAPIClient
from keyword_extractor import KeywordExtractor
from cache_manager import CacheManager
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app with static file configuration
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure static files
app.static_url_path = '/static'
app.static_folder = 'static'

# Swagger configuration
SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Research Paper Discovery System API",
        'dom_id': '#swagger-ui',
        'deepLinking': True,
        'supportedSubmitMethods': ['get'],
        'layout': 'BaseLayout',
        'docExpansion': 'list',
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Initialize components
try:
    cache_manager = CacheManager()
    keyword_extractor = KeywordExtractor()
    arxiv_client = ArxivAPIClient()
    paper_processor = PaperProcessor(keyword_extractor, arxiv_client, cache_manager)
    logger.info("All components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {str(e)}")
    raise

@app.route('/static/<path:path>')
def send_static(path):
    response = send_from_directory('static', path)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Content-Type'] = 'application/json' if path.endswith('.json') else response.headers.get('Content-Type')
    return response

@app.route('/api/research', methods=['GET'])
def get_research_papers():
    """Get research papers based on a question and optional criteria"""
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
        logger.error(f"API Error: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = 8080
    logger.info(f"Starting Flask API server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)