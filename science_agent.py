import logging
from typing import Dict, Any, List
from together import Together
import streamlit as st
import json
import time

class ScienceAgent:
    def __init__(self):
        self.model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        self.together_client = None
        self.rate_limit_delay = 10  # Increased to 10 seconds between requests
        self.last_request_time = 0
        if 'TOGETHER_API_KEY' in st.session_state:
            self.together_client = Together(api_key=st.session_state['TOGETHER_API_KEY'])
        logging.basicConfig(level=logging.INFO)

    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_request
            logging.info(f"Rate limiting: waiting {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def generate_hypothesis(self, paper_content: str, research_question: str) -> Dict[str, Any]:
        """Generate hypotheses based on paper content and research question"""
        try:
            if not self.together_client:
                raise ValueError("Together AI client not initialized")

            logging.info("🧪 Generating hypotheses...")
            self._rate_limit()  # Apply rate limiting

            response = self.together_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert scientific researcher. Based on the paper content and research question:
1. Generate potential hypotheses that could further the research
2. Identify gaps in current knowledge
3. Propose novel research directions
Respond with JSON in the format:
{
    "hypotheses": [
        {"hypothesis": "statement", "rationale": "explanation"}
    ],
    "knowledge_gaps": ["gap1", "gap2"],
    "research_directions": ["direction1", "direction2"]
}"""
                    },
                    {
                        "role": "user",
                        "content": f"Research Question: {research_question}\n\nPaper Content: {paper_content[:14000]}"  # Limit content length
                    }
                ],
                temperature=0.7,
                max_tokens=1000
            )

            content = response.choices[0].message.content.strip()
            result = self._parse_ai_response(content)

            # Ensure we have at least empty lists if parsing failed
            if 'hypotheses' not in result:
                result['hypotheses'] = []
            if 'knowledge_gaps' not in result:
                result['knowledge_gaps'] = []
            if 'research_directions' not in result:
                result['research_directions'] = []

            return result

        except Exception as e:
            logging.error(f"Error generating hypothesis: {str(e)}")
            return {
                "hypotheses": [
                    {
                        "hypothesis": "Unable to generate hypothesis due to API limitations",
                        "rationale": "The system encountered rate limiting or API errors. Please try again in a few minutes."
                    }
                ],
                "knowledge_gaps": ["Analysis limited due to API constraints"],
                "research_directions": ["Please try again after a brief waiting period"]
            }

    def design_experiments(self, hypothesis: str, context: str) -> Dict[str, Any]:
        """Design experiments to test the given hypothesis"""
        try:
            if not self.together_client:
                raise ValueError("Together AI client not initialized")

            logging.info("🔬 Designing experiments...")
            self._rate_limit()  # Apply rate limiting

            response = self.together_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert in experimental design. For the given hypothesis:
1. Design detailed experimental procedures
2. Specify required methodologies and techniques
3. Identify potential challenges and controls
Respond with JSON in the format:
{
    "experimental_design": {
        "overview": "brief description",
        "procedures": ["step1", "step2"],
        "methodologies": ["method1", "method2"],
        "required_equipment": ["equipment1", "equipment2"],
        "controls": ["control1", "control2"],
        "potential_challenges": ["challenge1", "challenge2"],
        "expected_outcomes": ["outcome1", "outcome2"]
    }
}"""
                    },
                    {
                        "role": "user",
                        "content": f"Hypothesis: {hypothesis}\n\nContext: {context[:14000]}"  # Limit context length
                    }
                ],
                temperature=0.7,
                max_tokens=1000
            )

            content = response.choices[0].message.content.strip()
            return self._parse_ai_response(content)

        except Exception as e:
            logging.error(f"Error designing experiments: {str(e)}")
            return {
                "experimental_design": {
                    "overview": "Error occurred during experimental design",
                    "procedures": ["Unable to generate procedures due to API limitations"],
                    "methodologies": ["Please try again after a brief waiting period"],
                    "required_equipment": [],
                    "controls": [],
                    "potential_challenges": ["API rate limiting encountered"],
                    "expected_outcomes": []
                }
            }

    def _parse_ai_response(self, content: str) -> Dict[str, Any]:
        """Parse and structure AI response"""
        try:
            if isinstance(content, str):
                return json.loads(content)
        except json.JSONDecodeError:
            logging.warning("Failed to parse JSON response, attempting manual structuring")

            # Extract content between code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass

            # Fallback parsing logic for non-JSON responses
            sections = {}
            current_section = None
            current_list = []

            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue

                if line.endswith(':'):
                    if current_section and current_list:
                        sections[current_section] = current_list
                    current_section = line[:-1].lower().replace(' ', '_')
                    current_list = []
                elif line.startswith(('- ', '* ', '• ')):
                    current_list.append(line[2:].strip())
                else:
                    current_list.append(line)

            if current_section and current_list:
                sections[current_section] = current_list

            return sections