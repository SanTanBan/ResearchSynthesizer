import os
import time
import logging
from typing import List, Dict, Any
from openai import OpenAI

class AbstractFilter:
    def __init__(self):
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.model = "gpt-4o"
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.rate_limit_delay = 1  # seconds between requests
        self.last_request_time = 0

    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last_request)
        self.last_request_time = time.time()

    def _analyze_abstract(self, abstract: str, research_question: str) -> bool:
        """Analyze a single abstract using GPT-4o"""
        try:
            self._rate_limit()
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert research paper analyst. "
                        "Determine if the given abstract is relevant to the research question. "
                        "Consider both direct relevance and potential indirect insights. "
                        "Respond with JSON containing: "
                        "{'is_relevant': boolean, 'confidence': float, 'reason': string}"
                    },
                    {
                        "role": "user",
                        "content": f"Research Question: {research_question}\n\nAbstract: {abstract}"
                    }
                ],
                response_format={"type": "json_object"}
            )

            result = response.choices[0].message.content
            analysis = eval(result)  # Safe since we requested JSON format
            
            # Consider paper relevant if confidence is high enough
            return analysis['is_relevant'] and analysis['confidence'] > 0.7

        except Exception as e:
            logging.error(f"Error analyzing abstract: {str(e)}")
            return True  # Include paper if analysis fails

    def filter_papers(self, papers: List[Dict[str, Any]], research_question: str) -> List[Dict[str, Any]]:
        """Filter papers based on abstract relevance"""
        try:
            filtered_papers = []
            total_papers = len(papers)
            
            logging.info(f"Starting to filter {total_papers} papers")
            
            for i, paper in enumerate(papers, 1):
                logging.info(f"Processing paper {i}/{total_papers}")
                
                if self._analyze_abstract(paper['abstract'], research_question):
                    filtered_papers.append(paper)
                
                # Log progress
                if i % 10 == 0:
                    logging.info(f"Processed {i}/{total_papers} papers. "
                               f"Kept {len(filtered_papers)} relevant papers so far.")

            logging.info(f"Filtering complete. Kept {len(filtered_papers)} out of {total_papers} papers")
            return filtered_papers

        except Exception as e:
            logging.error(f"Error in filter_papers: {str(e)}")
            return papers  # Return original list if filtering fails
