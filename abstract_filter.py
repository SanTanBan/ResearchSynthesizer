import os
import time
import logging
import json
from typing import List, Dict, Any
from openai import OpenAI
from together import Together
import streamlit as st

class AbstractFilter:
    def __init__(self):
        self.openai_model = "gpt-3.5-turbo-1106"  # Using faster GPT-3.5 model
        self.together_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        self.rate_limit_delay = 1  # seconds between requests
        self.last_request_time = 0
        self.setup_clients()

    def setup_clients(self):
        """Initialize API clients based on available credentials"""
        self.openai_client = None
        self.together_client = None

        if os.environ.get("OPENAI_API_KEY"):
            self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        if 'TOGETHER_API_KEY' in st.session_state:
            self.together_client = Together(api_key=st.session_state['TOGETHER_API_KEY'])

    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last_request)
        self.last_request_time = time.time()

    def _analyze_abstract_together(self, abstract: str, research_question: str) -> Dict[str, Any]:
        """Analyze a single abstract using Together AI"""
        try:
            if not self.together_client:
                raise ValueError("Together AI client not initialized")

            self._rate_limit()
            response = self.together_client.chat.completions.create(
                model=self.together_model,
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
                ]
            )

            content = response.choices[0].message.content
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Handle non-JSON response by parsing the text
                is_relevant = "relevant" in content.lower() and "not relevant" not in content.lower()
                confidence = 0.7 if is_relevant else 0.3
                return {
                    'is_relevant': is_relevant,
                    'confidence': confidence,
                    'reason': content
                }

            return {
                'is_relevant': result.get('is_relevant', True) and result.get('confidence', 0) > 1/3,
                'confidence': result.get('confidence', 0.7),
                'reason': result.get('reason', "No specific reason provided")
            }

        except Exception as e:
            logging.error(f"Error analyzing abstract with Together AI: {str(e)}")
            return None

    def _analyze_abstract(self, abstract: str, research_question: str) -> Dict[str, Any]:
        """Analyze a single abstract using available AI services with fallback"""
        # Try Together AI first
        if self.together_client:
            result = self._analyze_abstract_together(abstract, research_question)
            if result:
                return result

        # Fallback to OpenAI if available
        if self.openai_client:
            try:
                self._rate_limit()
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
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
                logging.error(f"Error analyzing abstract with OpenAI: {str(e)}")

        # Basic analysis as last resort
        return {
            'is_relevant': True,  # Include paper if no AI analysis is available
            'confidence': 0.5,
            'reason': "Basic inclusion: No AI service available for detailed analysis"
        }

    def filter_papers(self, papers: List[Dict[str, Any]], research_question: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Filter papers based on abstract relevance"""
        try:
            filtered_papers = []
            analysis_results = []
            total_papers = len(papers)

            logging.info(f"Starting to filter {total_papers} papers")

            for i, paper in enumerate(papers, 1):
                logging.info(f"Processing paper {i}/{total_papers}")

                analysis = self._analyze_abstract(paper['abstract'], research_question)
                paper['analysis'] = analysis

                if analysis['is_relevant']:
                    filtered_papers.append(paper)

                analysis_results.append({
                    'title': paper['title'],
                    'is_relevant': analysis['is_relevant'],
                    'confidence': analysis['confidence'],
                    'reason': analysis['reason']
                })

                if i % 10 == 0:
                    logging.info(f"Processed {i}/{total_papers} papers. "
                                f"Kept {len(filtered_papers)} relevant papers so far.")

            logging.info(f"Filtering complete. Kept {len(filtered_papers)} out of {total_papers} papers")
            return filtered_papers, analysis_results

        except Exception as e:
            logging.error(f"Error in filter_papers: {str(e)}")
            return papers, []  # Return original list if filtering fails