import arxiv
import time
from typing import List, Dict, Any
import logging

class ArxivAPIClient:
    def __init__(self):
        self.client = arxiv.Client()
        self.rate_limit_delay = 3  # seconds between requests
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last_request)
        self.last_request_time = time.time()
    
    def search_papers(self, keywords: List[str], max_results: int = 50) -> List[Dict[str, Any]]:
        """Search papers on arXiv based on keywords"""
        try:
            self._rate_limit()
            
            # Construct search query
            search_query = ' AND '.join(f'"{keyword}"' for keyword in keywords)
            
            # Create search parameters
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            # Execute search
            results = []
            for result in self.client.results(search):
                paper = {
                    'title': result.title,
                    'authors': [author.name for author in result.authors],
                    'abstract': result.summary,
                    'published': result.published.strftime("%Y-%m-%d"),
                    'arxiv_id': result.entry_id.split('/')[-1],
                    'url': result.pdf_url,
                    'categories': result.categories
                }
                results.append(paper)
            
            return results
            
        except Exception as e:
            logging.error(f"Error searching arXiv: {str(e)}")
            raise Exception(f"Failed to search arXiv: {str(e)}")
