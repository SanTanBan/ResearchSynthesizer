import os
import io
import logging
import requests
import json
from typing import List, Dict, Any
import PyPDF2
from openai import OpenAI

class PaperAnalyzer:
    def __init__(self):
        self.model = "gpt-3.5-turbo-1106"  # Using faster GPT-3.5 model
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def _download_pdf(self, arxiv_id: str) -> str:
        """Download PDF from arXiv and extract text"""
        try:
            # Construct PDF URL
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            logging.info(f"📥 Starting download of paper {arxiv_id} from: {pdf_url}")

            response = requests.get(pdf_url, timeout=30)  # Add timeout
            response.raise_for_status()
            logging.info(f"✅ Successfully downloaded PDF for paper {arxiv_id}")

            # Read PDF content
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            logging.info(f"📄 PDF loaded successfully for paper {arxiv_id}, pages: {len(pdf_reader.pages)}")

            # Extract text from each page (limit to first 20 pages for faster processing)
            text = ""
            for i, page in enumerate(pdf_reader.pages):
                if i >= 20:  # Only process first 20 pages
                    logging.info(f"⚡ Reached 20 page limit for paper {arxiv_id}, truncating...")
                    break
                text += page.extract_text() + "\n"
                if (i + 1) % 5 == 0:  # Log progress every 5 pages
                    logging.info(f"📝 Extracted text from {i + 1} pages for paper {arxiv_id}")

            if not text.strip():
                raise ValueError("No text extracted from PDF")

            logging.info(f"✨ Successfully extracted {len(text)} characters from paper {arxiv_id}")
            return text

        except Exception as e:
            logging.error(f"❌ Error processing PDF for {arxiv_id}: {str(e)}")
            return ""

    def _analyze_paper_content(self, paper_text: str, research_question: str) -> Dict[str, Any]:
        """Analyze a single paper using GPT-3.5"""
        try:
            # Truncate text if too long (token limit consideration)
            max_chars = 14000  # Approximate for GPT-3.5's context window
            truncated_text = paper_text[:max_chars] if len(paper_text) > max_chars else paper_text

            if len(paper_text) > max_chars:
                logging.info(f"📝 Truncated paper text from {len(paper_text)} to {len(truncated_text)} characters")

            logging.info("🤖 Starting GPT analysis of paper content...")

            # Enhanced prompt for better results
            response = self.openai_client.chat.completions.create(
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
                ],
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            logging.info("✅ Successfully completed GPT analysis")
            return result

        except Exception as e:
            logging.error(f"❌ Error analyzing paper content: {str(e)}")
            return {
                "error": str(e),
                "summary": "Error analyzing paper content",
                "relevant_points": []
            }

    def analyze_papers(self, papers: List[Dict[str, Any]], research_question: str) -> List[Dict[str, Any]]:
        """Analyze a list of papers and generate summaries"""
        paper_analyses = []
        total_papers = len(papers)

        for i, paper in enumerate(papers, 1):
            logging.info(f"\n🔍 Starting analysis of paper {i}/{total_papers}: {paper['title']}")
            logging.info("=" * 50)

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
                        'relevant_points': []
                    }
                })
                continue

            # Analyze content
            logging.info(f"📊 Beginning content analysis for paper: {paper['title']}")
            analysis = self._analyze_paper_content(paper_text, research_question)

            paper_analyses.append({
                'title': paper['title'],
                'arxiv_id': paper['arxiv_id'],
                'analysis': analysis
            })

            logging.info(f"✅ Completed analysis of paper {i}/{total_papers}: {paper['title']}")
            logging.info("-" * 50)

            # Add a small delay between papers
            if i < total_papers:
                import time
                time.sleep(2)  # Rate limiting

        logging.info(f"\n🎉 Completed analysis of all {total_papers} papers!")
        return paper_analyses

    def debug_with_sample_data(self, sample_papers: List[Dict[str, Any]], research_question: str) -> List[Dict[str, Any]]:
        """Debug method that accepts sample paper data"""
        return self.analyze_papers(sample_papers, research_question)