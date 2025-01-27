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

    def search_papers(self, keywords: List[str], max_results: int = 100) -> List[Dict[str, Any]]:
        """Search papers on arXiv based on keywords"""
        try:
            self._rate_limit()

            # Construct search query using OR between keywords
            query_parts = []
            for keyword in keywords:
                # Remove any special characters that might cause issues
                clean_keyword = keyword.replace('"', '').strip()
                if clean_keyword:
                    query_parts.append(f'"{clean_keyword}"')

            if not query_parts:
                raise ValueError("No valid keywords provided")

            # Combine with OR for broader results
            search_query = ' OR '.join(query_parts)

            # Create search parameters
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )

            logging.info(f"Searching arXiv with query: {search_query}")

            # Execute search with pagination
            results = []
            try:
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

                    # Apply rate limiting between results
                    if len(results) % 20 == 0:  # Rate limit every 20 papers
                        self._rate_limit()

                    if len(results) >= max_results:
                        break

            except Exception as e:
                # If we got some results before the error, return them
                if results:
                    logging.warning(f"Partial results retrieved due to: {str(e)}")
                else:
                    raise  # Re-raise if we got no results

            logging.info(f"Found {len(results)} papers for keywords: {keywords}")
            return results

        except Exception as e:
            logging.error(f"Error searching arXiv: {str(e)}")
            raise Exception(f"Failed to search arXiv: {str(e)}")