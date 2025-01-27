import os
from typing import List
from openai import OpenAI
import json
import logging

class KeywordExtractor:
    def __init__(self):
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.model = "gpt-4o"
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    def extract_keywords(self, research_question: str) -> List[str]:
        """Extract relevant keywords from research question using OpenAI"""
        try:
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
            
            return result.get('keywords', [])
            
        except Exception as e:
            logging.error(f"Error extracting keywords: {str(e)}")
            raise Exception(f"Failed to extract keywords: {str(e)}")
