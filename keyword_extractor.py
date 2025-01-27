import os
from typing import List
from openai import OpenAI
import json
import logging

class KeywordExtractor:
    def __init__(self):
        self.model = "gpt-3.5-turbo-1106"  # Using faster GPT-3.5 model
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.fallback_keywords = []

    def extract_keywords_basic(self, text: str) -> List[str]:
        """Basic keyword extraction without AI"""
        # Remove common words and punctuation
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = text.lower().replace('?', ' ').replace('!', ' ').replace('.', ' ').split()
        keywords = [word for word in words if word not in common_words and len(word) > 2]
        return list(set(keywords))[:7]  # Return up to 7 unique keywords

    def extract_keywords(self, research_question: str) -> List[str]:
        """Extract relevant keywords from research question using OpenAI with fallback"""
        try:
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("OpenAI API key is not set")

            response = self.openai_client.chat.completions.create(
                model=self.model,
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

            # Ensure we got a list of keywords
            if not isinstance(result.get('keywords', []), list):
                raise ValueError("Invalid response format from OpenAI")

            self.fallback_keywords = result.get('keywords', [])  # Store successful results
            return self.fallback_keywords

        except Exception as e:
            logging.error(f"Error extracting keywords: {str(e)}")
            # If we have previous successful keywords, use them
            if self.fallback_keywords:
                return self.fallback_keywords

            # Use basic keyword extraction as last resort
            basic_keywords = self.extract_keywords_basic(research_question)
            logging.info(f"Using basic keyword extraction. Keywords: {basic_keywords}")
            return basic_keywords