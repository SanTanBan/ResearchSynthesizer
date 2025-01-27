from typing import Dict, Any, List
import logging
from keyword_extractor import KeywordExtractor
from api_client import ArxivAPIClient
from cache_manager import CacheManager

class PaperProcessor:
    def __init__(self, 
                 keyword_extractor: KeywordExtractor,
                 arxiv_client: ArxivAPIClient,
                 cache_manager: CacheManager):
        self.keyword_extractor = keyword_extractor
        self.arxiv_client = arxiv_client
        self.cache_manager = cache_manager

    def _validate_input(self, research_question: str, criteria: str) -> None:
        """Validate input parameters"""
        if not research_question or not research_question.strip():
            raise ValueError("Research question cannot be empty")
        if len(research_question) > 1000:
            raise ValueError("Research question is too long (max 1000 characters)")

    def _filter_papers(self, papers: List[Dict[str, Any]], criteria: str) -> List[Dict[str, Any]]:
        """Filter papers based on criteria"""
        filtered_papers = papers

        if criteria:
            # Extract year constraint if present
            if "after" in criteria.lower():
                try:
                    year = int(''.join(filter(str.isdigit, criteria)))
                    filtered_papers = [
                        paper for paper in filtered_papers 
                        if int(paper['published'][:4]) >= year
                    ]
                except ValueError:
                    pass

            # Filter for specific study types if mentioned
            study_types = ['controlled trial', 'randomized', 'double-blind']
            for study_type in study_types:
                if study_type.lower() in criteria.lower():
                    filtered_papers = [
                        paper for paper in filtered_papers 
                        if study_type.lower() in paper['abstract'].lower()
                    ]

        return filtered_papers

    def process_query(self, research_question: str, criteria: str = "", max_papers: int = 20) -> Dict[str, Any]:
        """Process a research query and return relevant papers"""
        try:
            # Validate input
            self._validate_input(research_question, criteria)

            # Validate max_papers
            max_papers = max(3, min(49, max_papers))  # Ensure it's between 3 and 49

            # Check cache
            cache_key = f"{research_question}:{criteria}:{max_papers}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                logging.info("Using cached result")
                return cached_result

            # Extract keywords
            logging.info("Extracting keywords...")
            keywords = self.keyword_extractor.extract_keywords(research_question)
            logging.info(f"Extracted keywords: {keywords}")

            if not keywords:
                logging.error("No keywords extracted")
                return {
                    'papers': [],
                    'keywords': [],
                    'total_results': 0,
                    'error': 'No keywords could be extracted from the research question'
                }

            # Search papers with max_papers limit
            logging.info(f"Searching papers with keywords (limit: {max_papers})...")
            papers = self.arxiv_client.search_papers(keywords, max_papers)
            logging.info(f"Found {len(papers)} papers from search")

            # Filter papers
            filtered_papers = self._filter_papers(papers, criteria)
            logging.info(f"After filtering: {len(filtered_papers)} papers")

            # Prepare response
            result = {
                'papers': filtered_papers,
                'keywords': keywords,
                'total_results': len(filtered_papers)
            }

            # Cache result
            self.cache_manager.set(cache_key, result)

            return result

        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            raise