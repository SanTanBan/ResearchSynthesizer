import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from science_agent import ScienceAgent
from paper_analyzer import PaperAnalyzer
from abstract_filter import AbstractFilter

class ParallelProcessor:
    def __init__(self):
        self.science_agent = ScienceAgent()
        self.paper_analyzer = PaperAnalyzer()
        self.abstract_filter = AbstractFilter()
        self.max_workers = 4  # Limit concurrent processing
        logging.basicConfig(level=logging.INFO)

    def process_paper_pipeline(self, paper: Dict[str, Any], research_question: str) -> Dict[str, Any]:
        """Process a single paper through the complete pipeline"""
        try:
            paper_id = paper.get('arxiv_id', 'unknown')
            logging.info(f"🔄 Starting pipeline for paper {paper_id}")

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
                paper.get('full_text', paper['abstract']),  # Fallback to abstract if full text not available
                research_question
            )

            # Step 3: Hypothesis Generation
            hypotheses = self.science_agent.generate_hypothesis(
                paper.get('full_text', paper['abstract']),
                research_question
            )

            # Step 4: Experimental Design for each hypothesis
            experimental_designs = []
            for hypothesis in hypotheses.get('hypotheses', []):
                if isinstance(hypothesis, dict) and 'hypothesis' in hypothesis:
                    design = self.science_agent.design_experiments(
                        hypothesis['hypothesis'],
                        paper.get('full_text', paper['abstract'])
                    )
                    experimental_designs.append({
                        'hypothesis': hypothesis,
                        'design': design
                    })

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
            logging.error(f"❌ Error in pipeline for paper {paper_id}: {str(e)}")
            return {
                'paper_id': paper_id,
                'status': 'error',
                'error': str(e),
                'pipeline_results': None
            }

    def process_papers_parallel(self, papers: List[Dict[str, Any]], research_question: str) -> List[Dict[str, Any]]:
        """Process multiple papers in parallel"""
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_paper = {
                executor.submit(self.process_paper_pipeline, paper, research_question): paper 
                for paper in papers
            }

            for future in as_completed(future_to_paper):
                paper = future_to_paper[future]
                try:
                    result = future.result()
                    results.append(result)
                    logging.info(f"✅ Completed pipeline for paper {paper.get('arxiv_id', 'unknown')}")
                except Exception as e:
                    logging.error(f"❌ Pipeline failed for paper {paper.get('arxiv_id', 'unknown')}: {str(e)}")
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
