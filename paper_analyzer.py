import os
import io
import logging
import requests
import json
from typing import List, Dict, Any
import PyPDF2
from together import Together
import streamlit as st

class PaperAnalyzer:
    def __init__(self):
        self.model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        self.together_client = None
        if 'TOGETHER_API_KEY' in st.session_state:
            self.together_client = Together(api_key=st.session_state['TOGETHER_API_KEY'])
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - 📄 %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def _download_pdf(self, arxiv_id: str) -> str:
        """Download PDF from arXiv and extract text"""
        try:
            # Construct PDF URL
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            logging.info(f"⬇️ Downloading PDF for arXiv ID {arxiv_id}")

            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()

            # Read PDF content
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            logging.info(f"📚 Processing PDF: {len(pdf_reader.pages)} pages found")

            # Extract text from each page (limit to first 20 pages for faster processing)
            text = ""
            for i, page in enumerate(pdf_reader.pages):
                if i >= 20:  # Only process first 20 pages
                    logging.info("📝 Reached 20 page limit, truncating remaining pages")
                    break
                text += page.extract_text() + "\n"
                if (i + 1) % 5 == 0:  # Log progress every 5 pages
                    logging.info(f"📄 Processed {i + 1} pages...")

            if not text.strip():
                raise ValueError("No text extracted from PDF")

            logging.info(f"✅ Successfully extracted {len(text)} characters of text")
            return text

        except Exception as e:
            logging.error(f"❌ Error processing PDF {arxiv_id}: {str(e)}")
            return ""

    def _analyze_paper_content(self, paper_text: str, research_question: str) -> Dict[str, Any]:
        """Analyze paper content using Together AI"""
        try:
            if not self.together_client:
                raise ValueError("Together AI client not initialized")

            # Truncate text if too long
            max_chars = 14000  # Approximate context window
            truncated_text = paper_text[:max_chars] if len(paper_text) > max_chars else paper_text

            if len(paper_text) > max_chars:
                logging.info(f"📝 Truncating paper text from {len(paper_text)} to {max_chars} characters")

            logging.info("🤖 Sending paper to Together AI for analysis...")

            response = self.together_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a research paper analyzer. Your task is to:
1. Extract key points that specifically address the research question
2. Provide a concise summary focused on relevant findings
3. Highlight any limitations or caveats
Format your response as JSON with the following structure:
{
    "summary": "Brief overview of relevant findings",
    "relevant_points": ["Point 1", "Point 2", ...],
    "limitations": ["Limitation 1", "Limitation 2", ...]
}"""
                    },
                    {
                        "role": "user",
                        "content": f"Research Question: {research_question}\n\nPaper Content: {truncated_text}"
                    }
                ]
            )

            content = response.choices[0].message.content.strip()
            try:
                # Try to parse as JSON
                result = json.loads(content)
                logging.info("✅ Successfully analyzed paper content")
                return result
            except json.JSONDecodeError:
                # Handle non-JSON response by creating a structured response
                logging.warning("Non-JSON response received, structuring response manually")
                lines = content.split('\n')
                summary = ""
                points = []
                limitations = []
                current_section = None

                for line in lines:
                    line = line.strip()
                    if 'summary' in line.lower():
                        current_section = 'summary'
                    elif 'point' in line.lower() or 'finding' in line.lower():
                        current_section = 'points'
                    elif 'limitation' in line.lower():
                        current_section = 'limitations'
                    elif line:
                        if current_section == 'summary':
                            summary += line + " "
                        elif current_section == 'points':
                            points.append(line.lstrip('*-•').strip())
                        elif current_section == 'limitations':
                            limitations.append(line.lstrip('*-•').strip())

                return {
                    "summary": summary.strip(),
                    "relevant_points": points[:5],  # Limit to top 5 points
                    "limitations": limitations[:3]  # Limit to top 3 limitations
                }

        except Exception as e:
            logging.error(f"❌ Error analyzing paper content: {str(e)}")
            return {
                "error": str(e),
                "summary": "Error analyzing paper content",
                "relevant_points": [],
                "limitations": []
            }

    def analyze_papers(self, papers: List[Dict[str, Any]], research_question: str) -> List[Dict[str, Any]]:
        """Analyze a list of papers and generate summaries"""
        if not self.together_client:
            logging.error("Together AI client not initialized")
            return []

        paper_analyses = []
        total_papers = len(papers)

        logging.info(f"🚀 Starting analysis of {total_papers} papers")

        for i, paper in enumerate(papers, 1):
            logging.info(f"\n{'='*50}")
            logging.info(f"📊 Processing paper {i}/{total_papers}: {paper['title']}")
            logging.info(f"{'='*50}")

            # Get full text
            paper_text = self._download_pdf(paper['arxiv_id'])
            if not paper_text:
                logging.warning(f"⚠️ Could not extract text from paper {paper['arxiv_id']}")
                paper_analyses.append({
                    'title': paper['title'],
                    'arxiv_id': paper['arxiv_id'],
                    'analysis': {
                        'error': 'Could not extract text from PDF',
                        'summary': 'PDF extraction failed',
                        'relevant_points': [],
                        'limitations': []
                    }
                })
                continue

            # Analyze content
            analysis = self._analyze_paper_content(paper_text, research_question)

            paper_analyses.append({
                'title': paper['title'],
                'arxiv_id': paper['arxiv_id'],
                'analysis': analysis
            })

            # Add a small delay between papers
            if i < total_papers:
                import time
                time.sleep(2)  # Rate limiting
                logging.info(f"⏳ Waiting 2 seconds before processing next paper...")

        logging.info(f"\n✨ Completed analysis of all {total_papers} papers")
        return paper_analyses

    def debug_with_sample_data(self, sample_papers: List[Dict[str, Any]], research_question: str) -> List[Dict[str, Any]]:
        """Debug method that accepts sample paper data"""
        return self.analyze_papers(sample_papers, research_question)