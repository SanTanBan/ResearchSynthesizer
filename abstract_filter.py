import os
import time
import logging
import json
from typing import List, Dict, Any
from openai import OpenAI

class AbstractFilter:
    def __init__(self):
        self.model = "gpt-3.5-turbo-1106"  # Using faster GPT-3.5 model
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

    def _analyze_abstract(self, abstract: str, research_question: str) -> Dict[str, Any]:
        """Analyze a single abstract using GPT-3.5"""
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

            result = json.loads(response.choices[0].message.content)
            return {
                'is_relevant': result['is_relevant'] and result['confidence'] > 0.65,
                'confidence': result['confidence'],
                'reason': result['reason']
            }

        except Exception as e:
            logging.error(f"Error analyzing abstract: {str(e)}")
            return {
                'is_relevant': True,  # Include paper if analysis fails
                'confidence': 0.0,
                'reason': f"Error during analysis: {str(e)}"
            }

    def filter_papers(self, papers: List[Dict[str, Any]], research_question: str) -> List[Dict[str, Any]]:
        """Filter papers based on abstract relevance"""
        try:
            filtered_papers = []
            analysis_results = []  # Store analysis results for all papers
            total_papers = len(papers)

            logging.info(f"Starting to filter {total_papers} papers")

            for i, paper in enumerate(papers, 1):
                logging.info(f"Processing paper {i}/{total_papers}")

                analysis = self._analyze_abstract(paper['abstract'], research_question)
                paper['analysis'] = analysis  # Add analysis to paper data

                if analysis['is_relevant']:
                    filtered_papers.append(paper)

                analysis_results.append({
                    'title': paper['title'],
                    'is_relevant': analysis['is_relevant'],
                    'confidence': analysis['confidence'],
                    'reason': analysis['reason']
                })

                # Log progress
                if i % 10 == 0:
                    logging.info(f"Processed {i}/{total_papers} papers. "
                                f"Kept {len(filtered_papers)} relevant papers so far.")

            logging.info(f"Filtering complete. Kept {len(filtered_papers)} out of {total_papers} papers")
            return filtered_papers, analysis_results

        except Exception as e:
            logging.error(f"Error in filter_papers: {str(e)}")
            return papers, []  # Return original list if filtering fails