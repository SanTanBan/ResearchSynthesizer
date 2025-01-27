import os
from typing import List, Optional
from openai import OpenAI
import json
import logging
import random
from together import Together
from docx import Document
import streamlit as st

class KeywordExtractor:
    TOGETHER_MODELS = [
        "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "meta-llama/Llama-Vision-Free"
    ]

    def __init__(self):
        self.openai_model = "gpt-3.5-turbo-1106"
        self.together_model = self.TOGETHER_MODELS[0]  # Default model
        self.fallback_keywords = []
        self.setup_clients()

    def setup_clients(self):
        """Initialize API clients based on available credentials"""
        self.openai_client = None
        self.together_client = None

        if os.environ.get("OPENAI_API_KEY"):
            self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            logging.info("OpenAI client initialized")

        if 'TOGETHER_API_KEY' in st.session_state:
            self.together_client = Together(api_key=st.session_state['TOGETHER_API_KEY'])
            logging.info("Together AI client initialized")
        else:
            logging.warning("Together AI API key not found in session state")

    def set_together_model(self, model_name: str):
        """Set the Together AI model to use"""
        if model_name in self.TOGETHER_MODELS:
            self.together_model = model_name
            logging.info(f"Together AI model set to: {model_name}")

    def extract_keywords_together(self, research_question: str) -> List[str]:
        """Extract keywords using Together AI"""
        try:
            if not self.together_client:
                logging.error("Together AI client not initialized")
                raise ValueError("Together AI client not initialized")

            logging.info(f"Sending request to Together AI with model: {self.together_model}")
            response = self.together_client.chat.completions.create(
                model=self.together_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a research expert. Extract 5-7 relevant keywords from the research question. Return them as a JSON array of strings."
                    },
                    {"role": "user", "content": research_question}
                ]
            )

            # Parse the response and extract keywords
            content = response.choices[0].message.content
            logging.info(f"Together AI response: {content}")

            try:
                # Try to parse as pure JSON first
                keywords = json.loads(content)
                if isinstance(keywords, list):
                    logging.info(f"Successfully extracted keywords from JSON: {keywords}")
                    return keywords
                if isinstance(keywords, dict) and 'keywords' in keywords:
                    logging.info(f"Successfully extracted keywords from JSON dict: {keywords['keywords']}")
                    return keywords['keywords']
            except json.JSONDecodeError:
                # If not JSON, try to extract keywords from the text
                logging.info("Parsing non-JSON response")
                words = content.replace('[', '').replace(']', '').replace('"', '').split(',')
                keywords = [word.strip() for word in words if word.strip()]
                logging.info(f"Extracted keywords from text: {keywords}")
                return keywords

        except Exception as e:
            logging.error(f"Error extracting keywords with Together AI: {str(e)}")
            return None

    def extract_keywords(self, research_question: str) -> List[str]:
        """Extract relevant keywords with fallback mechanism"""
        logging.info("Starting keyword extraction")

        # Try Together AI first
        if self.together_client:
            logging.info("Attempting Together AI keyword extraction")
            keywords = self.extract_keywords_together(research_question)
            if keywords:
                self.fallback_keywords = keywords
                logging.info(f"Successfully extracted keywords using Together AI: {keywords}")
                return keywords
            logging.warning("Together AI keyword extraction failed")

        # Fallback to OpenAI
        try:
            if self.openai_client:
                logging.info("Attempting OpenAI keyword extraction")
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a research expert. Extract the most relevant "
                            "keywords from the given research question. Return them as a JSON "
                            "array of strings. Include only specific technical or scientific "
                            "terms that would be useful for searching academic papers. "
                            "Limit to 5-7 most relevant keywords."
                        },
                        {"role": "user", "content": research_question}
                    ],
                    response_format={"type": "json_object"}
                )

                result = json.loads(response.choices[0].message.content)
                if isinstance(result.get('keywords', []), list):
                    self.fallback_keywords = result.get('keywords', [])
                    logging.info(f"Successfully extracted keywords using OpenAI: {self.fallback_keywords}")
                    return self.fallback_keywords
                logging.warning("Invalid response format from OpenAI")

        except Exception as e:
            logging.error(f"Error extracting keywords with OpenAI: {str(e)}")

        # Use basic extraction as last resort
        if self.fallback_keywords:
            logging.info(f"Using fallback keywords: {self.fallback_keywords}")
            return self.fallback_keywords

        basic_keywords = self.extract_keywords_basic(research_question)
        logging.info(f"Using basic keyword extraction. Keywords: {basic_keywords}")
        return basic_keywords

    def extract_keywords_basic(self, text: str) -> List[str]:
        """Basic keyword extraction without AI"""
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = text.lower().replace('?', ' ').replace('!', ' ').replace('.', ' ').split()
        keywords = [word for word in words if word not in common_words and len(word) > 2]
        return list(set(keywords))[:7]