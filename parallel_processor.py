import logging
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from science_agent import ScienceAgent
from paper_analyzer import PaperAnalyzer
from abstract_filter import AbstractFilter

class ParallelProcessor:
    def __init__(self):
        self.science_agent = ScienceAgent()
        self.paper_analyzer = PaperAnalyzer()
        self.abstract_filter = AbstractFilter()
        self.max_workers = 3  # Reduced concurrent processing
        self.timeout = 300  # 5 minutes timeout for each paper
        logging.basicConfig(level=logging.INFO)

    def process_paper_pipeline(self, paper: Dict[str, Any], research_question: str) -> Dict[str, Any]:
        """Process a single paper through the complete pipeline"""
        max_retries = 3
        base_delay = 10  # Start with 10 seconds delay
        paper_id = paper.get('arxiv_id', 'unknown')
        title = paper.get('title', 'Unknown Title')

        for attempt in range(max_retries):
            try:
                logging.info(f"🔄 Starting pipeline for paper {paper_id}: {title} (Attempt {attempt + 1}/{max_retries})")

                # Step 1: Abstract Analysis
                abstract_analysis = self.abstract_filter._analyze_abstract(
                    paper['abstract'], 
                    research_question
                )

                if not abstract_analysis.get('is_relevant', False):
                    logging.info(f"❌ Paper {paper_id} marked as irrelevant, stopping pipeline")
                    return {
                        'paper_id': paper_id,
                        'status': 'filtered_out',
                        'reason': abstract_analysis.get('reason', 'Not relevant to research question'),
                        'pipeline_results': None
                    }

                # Step 2: Full Paper Analysis
                paper_analysis = self.paper_analyzer._analyze_paper_content(
                    paper.get('full_text', paper['abstract']),
                    research_question
                )

                if not paper_analysis or 'error' in paper_analysis:
                    raise Exception("Paper analysis failed")

                # Step 3: Hypothesis Generation
                hypotheses = self.science_agent.generate_hypothesis(
                    paper.get('full_text', paper['abstract']),
                    research_question
                )

                if not hypotheses or 'error' in hypotheses:
                    raise Exception("Hypothesis generation failed")

                # Step 4: Experimental Design
                experimental_designs = []
                for hypothesis in hypotheses.get('hypotheses', []):
                    if isinstance(hypothesis, dict) and 'hypothesis' in hypothesis:
                        design = self.science_agent.design_experiments(
                            hypothesis['hypothesis'],
                            paper.get('full_text', paper['abstract'])
                        )
                        if design and 'error' not in design:
                            experimental_designs.append({
                                'hypothesis': hypothesis,
                                'design': design
                            })

                # If we reach here, all steps completed successfully
                return {
                    'paper_id': paper_id,
                    'status': 'completed',
                    'pipeline_results': {
                        'paper_metadata': paper,
                        'abstract_analysis': abstract_analysis,
                        'paper_analysis': paper_analysis,
                        'hypotheses': hypotheses,
                        'experimental_designs': experimental_designs
                    }
                }

            except Exception as e:
                logging.error(f"❌ Error in pipeline for {paper_id} (Attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logging.info(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                    continue
                else:
                    return {
                        'paper_id': paper_id,
                        'status': 'error',
                        'error': str(e),
                        'pipeline_results': None
                    }

    def process_papers_parallel(self, papers: List[Dict[str, Any]], research_question: str) -> List[Dict[str, Any]]:
        """Process multiple papers in parallel with timeout"""
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_paper = {
                executor.submit(self.process_paper_pipeline, paper, research_question): paper 
                for paper in papers
            }

            completed = 0
            total = len(papers)

            for future in as_completed(future_to_paper):
                paper = future_to_paper[future]
                completed += 1

                try:
                    result = future.result(timeout=self.timeout)
                    results.append(result)
                    logging.info(f"✅ Completed pipeline for paper {completed}/{total}: {paper.get('title', 'Unknown')}")
                except TimeoutError:
                    logging.error(f"⏰ Timeout processing paper: {paper.get('title', 'Unknown')}")
                    results.append({
                        'paper_id': paper.get('arxiv_id', 'unknown'),
                        'status': 'timeout',
                        'error': 'Processing timed out',
                        'pipeline_results': None
                    })
                except Exception as e:
                    logging.error(f"❌ Pipeline failed for paper: {paper.get('title', 'Unknown')}: {str(e)}")
                    results.append({
                        'paper_id': paper.get('arxiv_id', 'unknown'),
                        'status': 'error',
                        'error': str(e),
                        'pipeline_results': None
                    })

        return results

    def aggregate_results(self, pipeline_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from all successful pipelines"""
        successful_results = [r for r in pipeline_results if r['status'] == 'completed']
        
        if not successful_results:
            return {
                'status': 'error',
                'message': 'No successful pipeline results to aggregate',
                'aggregated_data': None
            }

        # Aggregate all hypotheses and experimental designs
        all_hypotheses = []
        all_designs = []
        key_findings = []
        knowledge_gaps = set()

        for result in successful_results:
            pipeline_data = result['pipeline_results']
            if pipeline_data:
                # Collect hypotheses
                if 'hypotheses' in pipeline_data:
                    all_hypotheses.extend(pipeline_data['hypotheses'].get('hypotheses', []))
                    knowledge_gaps.update(pipeline_data['hypotheses'].get('knowledge_gaps', []))

                # Collect experimental designs
                if 'experimental_designs' in pipeline_data:
                    all_designs.extend(pipeline_data['experimental_designs'])

                # Collect key findings from paper analysis
                if 'paper_analysis' in pipeline_data:
                    key_findings.extend(pipeline_data['paper_analysis'].get('relevant_points', []))

        return {
            'status': 'success',
            'aggregated_data': {
                'total_papers_processed': len(pipeline_results),
                'successful_papers': len(successful_results),
                'key_findings': list(set(key_findings)),  # Remove duplicates
                'knowledge_gaps': list(knowledge_gaps),
                'proposed_hypotheses': all_hypotheses,
                'experimental_designs': all_designs
            }
        }