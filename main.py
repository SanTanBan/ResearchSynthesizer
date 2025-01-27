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

# Set page config before any other st commands
st.set_page_config(page_title="Novel Research Hypothesis Discovery Consolidation and Experimentation Design with Approach and Methodology", layout="wide")

# Update CSS styles
st.markdown("""
<style>
    .stTitle {
        font-size: 1.2rem !important;
        margin-bottom: 1rem !important;
    }
    .discovery-keywords {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .exp-design-header {
        font-size: 1.5rem !important;
        font-weight: bold !important;
        margin-top: 1.8rem !important;
        margin-bottom: 1.2rem !important;
    }
    .exp-design-subheader {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    .exp-design-bullet {
        margin-left: 2.5rem !important;
        margin-bottom: 0.7rem !important;
        font-size: 1.1rem !important;
    }
</style>
""", unsafe_allow_html=True)

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
    try:
        st.title("Novel Research Hypothesis Discovery Consolidation and Experimentation Design with Approach and Methodology")

        # Initialize session state for results
        if 'pipeline_results' not in st.session_state:
            st.session_state.pipeline_results = None
        if 'search_results' not in st.session_state:
            st.session_state.search_results = None

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
                min_value=0,
                max_value=25,  # Changed from 50 to 25
                value=5,  # Default value of 5 papers
                help="Set the maximum number of papers to search for (between 0 and 25)"
            )

            # Display Keywords (if available)
            if st.session_state.search_results:
                st.subheader("🔍 Search Keywords")
                st.write("Keywords used in search:")
                st.write(st.session_state.search_results.get('keywords', []))

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
            # Step 1: Initial Setup
            st.info("🚀 Starting paper discovery process...")
            if not 'TOGETHER_API_KEY' in st.session_state:
                st.error("⚠️ Together AI key is required for advanced analysis. Please upload your API key.")
                return

            # Step 2: Search Papers with max_papers limit
            with st.spinner("📚 Searching for relevant papers..."):
                results = paper_processor.process_query(research_question, criteria, max_papers)
                st.session_state.search_results = results
                initial_papers = len(results['papers'])

                # Display keywords in discovery process within an expander
                with st.expander("🔍 View Search Keywords", expanded=True):
                    st.markdown('<div class="discovery-keywords">', unsafe_allow_html=True)
                    st.write("The following keywords were extracted from your research question:")
                    for keyword in results['keywords']:
                        st.markdown(f"• {keyword}")
                    st.markdown('</div>', unsafe_allow_html=True)

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
                st.session_state.pipeline_results = pipeline_results

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

        # Display results if available in session state
        if st.session_state.pipeline_results:
            pipeline_results = st.session_state.pipeline_results

            # Step 4: Aggregate Results
            parallel_processor = ParallelProcessor()
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

                # Create hypothesis selection
                st.markdown("### 📌 Select a Hypothesis for Detailed Experimental Design")
                hypothesis_options = [f"Hypothesis {i+1}: {h['hypothesis']}" for i, h in enumerate(all_hypotheses)]
                if hypothesis_options:
                    selected_hypothesis_idx = st.selectbox(
                        "Choose a hypothesis to explore:",
                        range(len(hypothesis_options)),
                        format_func=lambda x: hypothesis_options[x]
                    )

                    # Display selected hypothesis details
                    selected_hypothesis = all_hypotheses[selected_hypothesis_idx]
                    st.markdown("### 🔍 Selected Hypothesis Details")
                    st.markdown("""
                    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
                    """, unsafe_allow_html=True)
                    st.write("**Hypothesis Statement:**")
                    st.write(selected_hypothesis.get('hypothesis', 'No hypothesis stated'))

                    st.write("**Scientific Rationale:**")
                    st.write(selected_hypothesis.get('rationale', 'No rationale provided'))

                    if 'supporting_evidence' in selected_hypothesis:
                        st.write("**Supporting Evidence:**")
                        for evidence in selected_hypothesis['supporting_evidence']:
                            st.write(f"• {evidence}")

                    if 'potential_impact' in selected_hypothesis:
                        st.write("**Potential Impact:**")
                        st.write(selected_hypothesis.get('potential_impact', ''))
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Find experimental designs for selected hypothesis
                    st.markdown("### 🧪 Experimental Design")
                    related_designs = []
                    for result in pipeline_results:
                        if result['status'] == 'completed':
                            for design in result['pipeline_results'].get('experimental_designs', []):
                                if design.get('hypothesis', {}).get('hypothesis') == selected_hypothesis.get('hypothesis'):
                                    related_designs.append(design['design'])

                    if related_designs:
                        for j, design in enumerate(related_designs, 1):
                            if isinstance(design, dict) and 'experimental_design' in design:
                                exp_design = design['experimental_design']
                                with st.expander(f"Experimental Design Variation {j}"):
                                    st.markdown('<p class="exp-design-header">Overview</p>', unsafe_allow_html=True)
                                    st.write(exp_design.get('overview', ''))

                                    if exp_design.get('procedures'):
                                        st.markdown('<p class="exp-design-subheader">Detailed Procedures</p>', unsafe_allow_html=True)
                                        for proc in exp_design['procedures']:
                                            st.markdown(f'<p class="exp-design-bullet">• {proc}</p>', unsafe_allow_html=True)

                                    if exp_design.get('methodologies'):
                                        st.markdown('<p class="exp-design-subheader">Required Methodologies</p>', unsafe_allow_html=True)
                                        for method in exp_design['methodologies']:
                                            st.markdown(f'<p class="exp-design-bullet">• {method}</p>', unsafe_allow_html=True)

                                    if exp_design.get('required_equipment'):
                                        st.markdown('<p class="exp-design-subheader">Required Equipment and Resources</p>', unsafe_allow_html=True)
                                        for equip in exp_design['required_equipment']:
                                            st.markdown(f'<p class="exp-design-bullet">• {equip}</p>', unsafe_allow_html=True)

                                    if exp_design.get('controls'):
                                        st.markdown('<p class="exp-design-subheader">Control Measures</p>', unsafe_allow_html=True)
                                        for control in exp_design['controls']:
                                            st.markdown(f'<p class="exp-design-bullet">• {control}</p>', unsafe_allow_html=True)

                                    if exp_design.get('potential_challenges'):
                                        st.markdown('<p class="exp-design-subheader">Potential Challenges and Mitigation Strategies</p>', unsafe_allow_html=True)
                                        for challenge in exp_design['potential_challenges']:
                                            st.markdown(f'<p class="exp-design-bullet">• {challenge}</p>', unsafe_allow_html=True)

                                    if exp_design.get('expected_outcomes'):
                                        st.markdown('<p class="exp-design-subheader">Expected Outcomes and Success Metrics</p>', unsafe_allow_html=True)
                                        for outcome in exp_design['expected_outcomes']:
                                            st.markdown(f'<p class="exp-design-bullet">• {outcome}</p>', unsafe_allow_html=True)

                                    if exp_design.get('novelty_analysis'):
                                        st.markdown('<p class="exp-design-subheader">Research Novelty Analysis</p>', unsafe_allow_html=True)
                                        for point in exp_design['novelty_analysis']:
                                            st.markdown(f'<p class="exp-design-bullet">• {point}</p>', unsafe_allow_html=True)

                                    if exp_design.get('bioinformatics_tools'):
                                        st.markdown('<p class="exp-design-subheader">Required Bioinformatics Tools and Datasets</p>', unsafe_allow_html=True)
                                        for tool in exp_design['bioinformatics_tools']:
                                            st.markdown(f'<p class="exp-design-bullet">• {tool}</p>', unsafe_allow_html=True)

                    else:
                        st.warning("No hypotheses generated from the analysis.")

            else:
                st.error("❌ No successful pipeline results to display")

    except Exception as e:
        st.error(f"An error occurred while processing your request: {str(e)}")
        logging.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()