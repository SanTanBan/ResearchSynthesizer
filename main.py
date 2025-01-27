import os
import logging
import streamlit as st
import json
from paper_processor import PaperProcessor
from api_client import ArxivAPIClient
from keyword_extractor import KeywordExtractor
from cache_manager import CacheManager
from abstract_filter import AbstractFilter
from paper_analyzer import PaperAnalyzer
from parallel_processor import ParallelProcessor
from science_agent import ScienceAgent

st.set_page_config(page_title="Research Paper Discovery System", layout="wide")

def handle_together_api_key(uploaded_file):
    """Process uploaded Together AI API key file"""
    if uploaded_file is not None:
        try:
            # Read the docx file
            doc_bytes = uploaded_file.read()
            with open("temp.docx", "wb") as f:
                f.write(doc_bytes)
            from docx import Document
            doc = Document("temp.docx")
            api_key = " ".join(paragraph.text for paragraph in doc.paragraphs)
            os.remove("temp.docx")  # Clean up
            st.session_state['TOGETHER_API_KEY'] = api_key
            return True
        except Exception as e:
            st.error(f"Error processing API key file: {str(e)}")
            return False
    return False

def main():
    st.title("Research Paper Discovery System")

    # Sidebar for configuration
    with st.sidebar:
        st.subheader("🔑 API Configuration")

        # Together AI API Key Upload
        st.markdown("""
        #### Upload Together AI API Key
        Please upload a .docx file containing **only** your Together AI API key.
        The file should contain nothing else but the API key text.
        """)
        uploaded_file = st.file_uploader("Upload API Key (docx)", type=['docx'])
        if uploaded_file:
            if handle_together_api_key(uploaded_file):
                st.success("Together AI API key loaded successfully!")

        # Model Selection
        st.subheader("🤖 Model Configuration")
        selected_model = None
        if 'TOGETHER_API_KEY' in st.session_state:
            models = KeywordExtractor.TOGETHER_MODELS
            use_together = st.checkbox("Use Together AI", value=True)
            if use_together:
                selected_model = st.selectbox("Select Together AI Model", models)

        # Paper Limit Control
        st.subheader("📚 Search Configuration")
        max_papers = st.slider(
            "Maximum Papers to Search",
            min_value=3,
            max_value=49,
            value=20,
            help="Set the maximum number of papers to search for (between 3 and 49)"
        )

    # Initialize components
    cache_manager = CacheManager()
    keyword_extractor = KeywordExtractor()
    if selected_model:
        keyword_extractor.set_together_model(selected_model)

    arxiv_client = ArxivAPIClient()
    paper_processor = PaperProcessor(keyword_extractor, arxiv_client, cache_manager)
    abstract_filter = AbstractFilter()

    # Input form
    with st.form("research_query_form"):
        research_question = st.text_area(
            "Research Question/Hypothesis",
            help="Enter your research question or hypothesis"
        )

        criteria = st.text_area(
            "Additional Criteria",
            help="Enter additional criteria (e.g., publication quality, study type)",
            placeholder="Example: only double-blind controlled trials, published after 2020"
        )

        submitted = st.form_submit_button("Search Papers")

    if submitted:
        try:
            # Step 1: Initial Setup
            st.info("🚀 Starting paper discovery process...")
            if not 'TOGETHER_API_KEY' in st.session_state:
                st.error("⚠️ Together AI key is required for advanced analysis. Please upload your API key.")
                return

            # Step 2: Search Papers with max_papers limit
            with st.spinner("📚 Searching for relevant papers..."):
                results = paper_processor.process_query(research_question, criteria, max_papers)
                initial_papers = len(results['papers'])
                st.success(f"Found {initial_papers} papers based on keyword search.")

            # Step 3: Initialize Parallel Processing
            st.write("🔄 Starting parallel analysis pipelines...")
            parallel_processor = ParallelProcessor()

            # Create a progress container
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Process papers in parallel
                pipeline_results = parallel_processor.process_papers_parallel(
                    results['papers'],
                    research_question
                )

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

            # Step 4: Aggregate Results
            st.write("🔄 Aggregating results from all pipelines...")
            aggregated_results = parallel_processor.aggregate_results(pipeline_results)

            if aggregated_results['status'] == 'success':
                agg_data = aggregated_results['aggregated_data']

                # Display Summary Statistics
                st.subheader("📊 Analysis Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Papers Processed", agg_data['total_papers_processed'])
                with col2:
                    st.metric("Successfully Analyzed", agg_data['successful_papers'])

                # Display Key Findings
                if agg_data['key_findings']:
                    st.subheader("🔑 Key Findings")
                    for finding in agg_data['key_findings']:
                        st.write(f"• {finding}")

                # Display Knowledge Gaps
                if agg_data['knowledge_gaps']:
                    st.subheader("🔍 Knowledge Gaps Identified")
                    for gap in agg_data['knowledge_gaps']:
                        st.write(f"• {gap}")

                # Display Hypotheses and Experimental Designs
                st.subheader("🧪 Generated Hypotheses and Experimental Designs")
                for i, hypothesis in enumerate(agg_data['proposed_hypotheses'], 1):
                    with st.expander(f"Hypothesis {i}: {hypothesis.get('hypothesis', 'Unknown')}"):
                        st.write("**Rationale:**")
                        st.write(hypothesis.get('rationale', 'No rationale provided'))

                        # Find matching experimental design
                        matching_designs = [
                            design for design in agg_data['experimental_designs']
                            if design.get('hypothesis', {}).get('hypothesis') == hypothesis.get('hypothesis')
                        ]

                        if matching_designs:
                            st.write("**Experimental Design:**")
                            design = matching_designs[0]['design']
                            if isinstance(design, dict) and 'experimental_design' in design:
                                exp_design = design['experimental_design']
                                st.write("Overview:", exp_design.get('overview', ''))

                                st.write("**Procedures:**")
                                for proc in exp_design.get('procedures', []):
                                    st.write(f"• {proc}")

                                st.write("**Required Equipment:**")
                                for equip in exp_design.get('required_equipment', []):
                                    st.write(f"• {equip}")

                                st.write("**Potential Challenges:**")
                                for challenge in exp_design.get('potential_challenges', []):
                                    st.write(f"• {challenge}")

                # Display Individual Paper Results
                st.subheader("📑 Individual Paper Results")
                for result in pipeline_results:
                    if result['status'] == 'completed':
                        with st.expander(f"📄 {result['pipeline_results']['paper_metadata']['title']}"):
                            st.write("**Status:** ✅ Complete")
                            if 'paper_analysis' in result['pipeline_results']:
                                st.write("**Summary:**")
                                st.write(result['pipeline_results']['paper_analysis'].get('summary', ''))
                    else:
                        with st.expander(f"📄 Paper ID: {result['paper_id']}"):
                            st.write(f"**Status:** ❌ {result['status']}")
                            if 'error' in result:
                                st.error(f"Error: {result['error']}")

            else:
                st.error("❌ No successful pipeline results to display")

        except Exception as e:
            st.error(f"An error occurred while processing your request: {str(e)}")
            logging.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()