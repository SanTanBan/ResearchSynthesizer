import logging
from typing import Dict, Any, List
from together import Together
import streamlit as st
import json

class ScienceAgent:
    def __init__(self):
        self.model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        self.together_client = None
        if 'TOGETHER_API_KEY' in st.session_state:
            self.together_client = Together(api_key=st.session_state['TOGETHER_API_KEY'])

        logging.basicConfig(level=logging.INFO)

    def generate_hypothesis(self, paper_content: str, research_question: str) -> Dict[str, Any]:
        """Generate hypotheses based on paper content and research question"""
        try:
            if not self.together_client:
                raise ValueError("Together AI client not initialized")

            logging.info("🧪 Generating hypotheses...")
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
                        "content": f"Research Question: {research_question}\n\nPaper Content: {paper_content}"
                    }
                ]
            )

            content = response.choices[0].message.content.strip()
            return self._parse_ai_response(content)
        except Exception as e:
            logging.error(f"Error generating hypothesis: {str(e)}")
            return {
                "error": str(e),
                "hypotheses": [],
                "knowledge_gaps": [],
                "research_directions": []
            }

    def design_experiments(self, hypothesis: str, context: str) -> Dict[str, Any]:
        """Design experiments to test the given hypothesis"""
        try:
            if not self.together_client:
                raise ValueError("Together AI client not initialized")

            logging.info("🔬 Designing experiments...")
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
                        "content": f"Hypothesis: {hypothesis}\n\nContext: {context}"
                    }
                ]
            )

            content = response.choices[0].message.content.strip()
            return self._parse_ai_response(content)
        except Exception as e:
            logging.error(f"Error designing experiments: {str(e)}")
            return {
                "error": str(e),
                "experimental_design": {
                    "overview": "Error occurred during experimental design",
                    "procedures": [],
                    "methodologies": [],
                    "required_equipment": [],
                    "controls": [],
                    "potential_challenges": [],
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