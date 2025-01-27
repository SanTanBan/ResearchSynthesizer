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

st.set_page_config(page_title="Novel Research Hypothesis Discovery Consolidation and Experimentation Design with Approach and Methodology", layout="wide")

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
    st.title("Novel Research Hypothesis Discovery Consolidation and Experimentation Design with Approach and Methodology")

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

                # Display Initial Paper Analysis Results
                st.subheader("📑 Paper Analysis Results")
                for result in pipeline_results:
                    status_icon = "✅" if result['status'] == 'completed' else "❌"
                    status_color = "green" if result['status'] == 'completed' else "red"

                    with st.expander(f"{status_icon} Paper: {result.get('pipeline_results', {}).get('paper_metadata', {}).get('title', 'Unknown Title')}"):
                        if result['status'] == 'completed':
                            paper_data = result['pipeline_results']['paper_metadata']
                            st.markdown(f"**Status:** :green[Selected for Analysis]")

                            # Paper Details
                            st.markdown("### 📝 Paper Details")
                            st.write(f"**Authors:** {', '.join(paper_data.get('authors', []))}")
                            st.write(f"**Published:** {paper_data.get('published', 'N/A')}")
                            st.write(f"**arXiv ID:** {paper_data.get('arxiv_id', 'N/A')}")
                            st.write(f"**URL:** {paper_data.get('url', 'N/A')}")

                            # Abstract and Analysis
                            st.markdown("### 📌 Abstract")
                            st.write(paper_data.get('abstract', 'No abstract available'))

                            if 'abstract_analysis' in result['pipeline_results']:
                                analysis = result['pipeline_results']['abstract_analysis']
                                st.markdown("### 🔍 Relevance Analysis")
                                st.write(f"**Confidence Score:** {analysis.get('confidence', 0):.2f}")
                                st.write(f"**Reason for Selection:** {analysis.get('reason', 'No reason provided')}")

                            if 'paper_analysis' in result['pipeline_results']:
                                st.markdown("### 📊 Detailed Analysis")
                                paper_analysis = result['pipeline_results']['paper_analysis']
                                st.write("**Summary:**")
                                st.write(paper_analysis.get('summary', 'No summary available'))

                                if paper_analysis.get('relevant_points'):
                                    st.write("**Key Points:**")
                                    for point in paper_analysis['relevant_points']:
                                        st.write(f"• {point}")
                        else:
                            st.markdown(f"**Status:** :red[Not Selected]")
                            st.write(f"**Reason:** {result.get('error', 'Unknown error')}")

                # Display Consolidated Hypotheses
                st.subheader("🧪 Consolidated Hypotheses and Experimental Designs")

                # Group similar hypotheses
                all_hypotheses = []
                for result in pipeline_results:
                    if result['status'] == 'completed':
                        pipeline_data = result['pipeline_results']
                        if 'hypotheses' in pipeline_data:
                            all_hypotheses.extend(pipeline_data['hypotheses'].get('hypotheses', []))

                # Display consolidated hypotheses with their experimental designs
                for i, hypothesis in enumerate(all_hypotheses, 1):
                    with st.expander(f"Hypothesis {i}: {hypothesis.get('hypothesis', 'Unknown')}"):
                        st.write("**Hypothesis Statement:**")
                        st.write(hypothesis.get('hypothesis', 'No hypothesis stated'))

                        st.write("**Rationale:**")
                        st.write(hypothesis.get('rationale', 'No rationale provided'))

                        # Find all experimental designs for this hypothesis
                        related_designs = []
                        for result in pipeline_results:
                            if result['status'] == 'completed':
                                for design in result['pipeline_results'].get('experimental_designs', []):
                                    if design.get('hypothesis', {}).get('hypothesis') == hypothesis.get('hypothesis'):
                                        related_designs.append(design['design'])

                        if related_designs:
                            st.markdown("### 🔬 Experimental Design")
                            for j, design in enumerate(related_designs, 1):
                                if isinstance(design, dict) and 'experimental_design' in design:
                                    exp_design = design['experimental_design']
                                    st.write(f"**Design Variation {j}:**")
                                    st.write("Overview:", exp_design.get('overview', ''))

                                    if exp_design.get('procedures'):
                                        st.write("**Procedures:**")
                                        for proc in exp_design['procedures']:
                                            st.write(f"• {proc}")

                                    if exp_design.get('methodologies'):
                                        st.write("**Methodologies:**")
                                        for method in exp_design['methodologies']:
                                            st.write(f"• {method}")

                                    if exp_design.get('required_equipment'):
                                        st.write("**Required Equipment:**")
                                        for equip in exp_design['required_equipment']:
                                            st.write(f"• {equip}")

                                    if exp_design.get('potential_challenges'):
                                        st.write("**Potential Challenges:**")
                                        for challenge in exp_design['potential_challenges']:
                                            st.write(f"• {challenge}")

            else:
                st.error("❌ No successful pipeline results to display")

        except Exception as e:
            st.error(f"An error occurred while processing your request: {str(e)}")
            logging.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()