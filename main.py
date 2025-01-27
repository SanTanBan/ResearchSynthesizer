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
            # Step 1: Initial Setup
            st.info("🚀 Starting paper discovery process...")
            if not os.environ.get("OPENAI_API_KEY"):
                st.warning("⚠️ OpenAI API key is not set. Using basic keyword extraction.")

            # Step 2: Search Papers
            with st.spinner("📚 Searching for relevant papers..."):
                results = paper_processor.process_query(research_question, criteria)
                initial_papers = len(results['papers'])
                st.success(f"Found {initial_papers} papers based on keyword search.")

            # Step 3: Filter Papers
            st.write("🤖 Analyzing paper abstracts for relevance...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            filter_results = st.empty()

            papers_before = len(results['papers'])

            # Create two columns for showing kept and dropped papers
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📋 Kept Papers")
                kept_container = st.container()

            with col2:
                st.subheader("❌ Dropped Papers")
                dropped_container = st.container()

            # Filter papers with progress updates
            filtered_papers, analysis_results = abstract_filter.filter_papers(
                results['papers'], 
                research_question
            )

            # Update progress bar and status for each paper
            for i, result in enumerate(analysis_results):
                progress = (i + 1) / len(analysis_results)
                progress_bar.progress(progress)
                status_text.text(f"Analyzing paper {i + 1} of {len(analysis_results)}")

                # Show decision in respective column
                if result['is_relevant']:
                    with kept_container:
                        st.write(f"**{result['title']}**")
                        st.write(f"Confidence: {result['confidence']:.2f}")
                        st.write(f"Reason: {result['reason']}")
                        st.divider()
                else:
                    with dropped_container:
                        st.write(f"**{result['title']}**")
                        st.write(f"Confidence: {result['confidence']:.2f}")
                        st.write(f"Reason: {result['reason']}")
                        st.divider()

            papers_after = len(filtered_papers)
            progress_bar.empty()
            status_text.empty()

            st.success(f"✨ Analysis complete! Kept {papers_after} out of {papers_before} papers.")

            # Display final results
            st.sidebar.subheader("📊 Analysis Summary")
            st.sidebar.markdown(f"**Initial papers found:** {papers_before}")
            st.sidebar.markdown(f"**Papers after filtering:** {papers_after}")
            st.sidebar.markdown("**Keywords used:**")
            st.sidebar.write(results['keywords'])

            if 'quota_exceeded' in results:
                st.warning("⚠️ OpenAI API quota exceeded. Using basic keyword extraction.")

            # Display papers
            st.subheader("📑 Final Relevant Papers")
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