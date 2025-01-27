import os
import logging
import streamlit as st
import json
from paper_processor import PaperProcessor
from api_client import ArxivAPIClient
from keyword_extractor import KeywordExtractor
from cache_manager import CacheManager
from abstract_filter import AbstractFilter

st.set_page_config(page_title="Research Paper Discovery System", layout="wide")

def main():
    st.title("Research Paper Discovery System")

    # Initialize components
    cache_manager = CacheManager()
    keyword_extractor = KeywordExtractor()
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
            # Create a collapsible summary section
            with st.expander("🔍 Filtering Process Summary", expanded=True):
                summary_placeholder = st.empty()
                progress_bar = st.progress(0)
                status_text = st.empty()

            # Step 1: Initial Setup
            if not os.environ.get("OPENAI_API_KEY"):
                st.warning("⚠️ OpenAI API key is not set. Using basic keyword extraction.")

            # Step 2: Search Papers
            with st.spinner("📚 Searching for relevant papers..."):
                results = paper_processor.process_query(research_question, criteria)
                initial_papers = len(results['papers'])

                # Update summary
                summary_content = f"""
                ### Current Progress
                1. **Initial Search** ✓
                   - Found {initial_papers} papers
                   - Keywords: {', '.join(results['keywords'])}
                2. **Abstract Analysis** 🔄
                   - Papers to analyze: {initial_papers}
                   - Progress: Starting...
                """
                summary_placeholder.markdown(summary_content)

            # Add CSS to create scrollable containers
            st.markdown("""
                <style>
                    .scrollable-container {
                        height: 400px;
                        overflow-y: auto;
                        padding: 1rem;
                        border: 1px solid #ddd;
                        border-radius: 0.5rem;
                        background-color: #ffffff;
                        margin-bottom: 1rem;
                    }
                </style>
            """, unsafe_allow_html=True)

            # Create collapsible section for detailed analysis
            with st.expander("📊 Detailed Analysis View", expanded=False):
                # Create two columns for showing kept and dropped papers
                col1, col2 = st.columns(2)

                # Initialize containers for kept and dropped papers
                with col1:
                    st.subheader("📋 Kept Papers")
                    st.markdown('<div class="scrollable-container" id="kept-papers">', unsafe_allow_html=True)
                    kept_papers_container = st.empty()
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.subheader("❌ Dropped Papers")
                    st.markdown('<div class="scrollable-container" id="dropped-papers">', unsafe_allow_html=True)
                    dropped_papers_container = st.empty()
                    st.markdown('</div>', unsafe_allow_html=True)

            # Filter papers with progress updates
            filtered_papers, analysis_results = abstract_filter.filter_papers(
                results['papers'], 
                research_question
            )

            # Initialize content lists
            kept_papers = []
            dropped_papers = []

            # Update progress bar and status for each paper
            for i, result in enumerate(analysis_results):
                progress = (i + 1) / len(analysis_results)
                progress_bar.progress(progress)

                # Update status and summary
                status_text.text(f"Analyzing paper {i + 1} of {len(analysis_results)}")
                current_kept = sum(1 for r in analysis_results[:i+1] if r['is_relevant'])
                summary_content = f"""
                ### Current Progress
                1. **Initial Search** ✓
                   - Found {initial_papers} papers
                   - Keywords: {', '.join(results['keywords'])}
                2. **Abstract Analysis** 🔄
                   - Papers analyzed: {i + 1} of {len(analysis_results)}
                   - Papers kept so far: {current_kept}
                   - Progress: {progress:.0%} complete
                """
                summary_placeholder.markdown(summary_content)

                # Format paper info
                paper_info = f"""
                <div style='margin-bottom: 1rem;'>
                    <h4>{result['title']}</h4>
                    <p>Confidence: {result['confidence']:.2f}</p>
                    <p>Reason: {result['reason']}</p>
                    <hr>
                </div>
                """

                # Add to respective list
                if result['is_relevant']:
                    kept_papers.append(paper_info)
                else:
                    dropped_papers.append(paper_info)

                # Update containers with HTML content
                if i % 5 == 0 or i == len(analysis_results) - 1:
                    kept_papers_container.markdown('\n'.join(kept_papers), unsafe_allow_html=True)
                    dropped_papers_container.markdown('\n'.join(dropped_papers), unsafe_allow_html=True)

            papers_after = len(filtered_papers)
            progress_bar.empty()
            status_text.empty()

            # Final summary update
            final_summary = f"""
            ### Final Results
            1. **Initial Search** ✓
               - Found {initial_papers} papers
               - Keywords: {', '.join(results['keywords'])}
            2. **Abstract Analysis** ✓
               - Papers analyzed: {len(analysis_results)}
               - Papers kept: {papers_after}
               - Papers filtered out: {initial_papers - papers_after}
               - Success rate: {(papers_after/initial_papers*100):.1f}%
            """
            summary_placeholder.markdown(final_summary)

            # Display sidebar summary
            st.sidebar.subheader("📊 Analysis Summary")
            st.sidebar.markdown(f"**Initial papers found:** {initial_papers}")
            st.sidebar.markdown(f"**Papers after filtering:** {papers_after}")
            st.sidebar.markdown("**Keywords used:**")
            st.sidebar.write(results['keywords'])

            if 'quota_exceeded' in results:
                st.warning("⚠️ OpenAI API quota exceeded. Using basic keyword extraction.")

            # Display final papers in a collapsible section
            with st.expander("📑 Final Relevant Papers", expanded=True):
                for paper in filtered_papers:
                    with st.expander(f"📄 {paper['title']}"):
                        st.write(f"**Authors:** {', '.join(paper['authors'])}")
                        st.write(f"**Published:** {paper['published']}")
                        st.write(f"**Abstract:** {paper['abstract']}")
                        st.write(f"**arXiv ID:** {paper['arxiv_id']}")
                        st.write(f"**URL:** {paper['url']}")
                        if 'analysis' in paper:
                            st.write(f"**Relevance Score:** {paper['analysis']['confidence']:.2f}")
                            st.write(f"**Reason for inclusion:** {paper['analysis']['reason']}")

        except Exception as e:
            st.error(f"An error occurred while processing your request. Please try again later.")
            logging.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()