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

        use_filter = st.checkbox(
            "Use AI to filter relevant papers",
            help="Use GPT-4o to analyze abstracts and keep only relevant papers"
        )

        submitted = st.form_submit_button("Search Papers")

    if submitted:
        try:
            with st.spinner("Processing your request..."):
                if not os.environ.get("OPENAI_API_KEY"):
                    st.warning("⚠️ OpenAI API key is not set. Using basic keyword extraction.")

                results = paper_processor.process_query(research_question, criteria)

                if use_filter:
                    with st.spinner("Analyzing abstracts for relevance..."):
                        papers_before = len(results['papers'])
                        results['papers'] = abstract_filter.filter_papers(
                            results['papers'], 
                            research_question
                        )
                        papers_after = len(results['papers'])
                        st.info(f"AI filter kept {papers_after} relevant papers out of {papers_before} papers.")

                # Display keywords first
                st.sidebar.subheader("Extracted Keywords")
                st.sidebar.write(results['keywords'])
                if 'quota_exceeded' in results:
                    st.warning("⚠️ OpenAI API quota exceeded. Using basic keyword extraction.")

                # Display results
                st.subheader(f"Found {len(results['papers'])} Papers")
                for paper in results['papers']:
                    with st.expander(f"{paper['title']}"):
                        st.write(f"**Authors:** {', '.join(paper['authors'])}")
                        st.write(f"**Published:** {paper['published']}")
                        st.write(f"**Abstract:** {paper['abstract']}")
                        st.write(f"**arXiv ID:** {paper['arxiv_id']}")
                        st.write(f"**URL:** {paper['url']}")

        except Exception as e:
            st.error(f"An error occurred while processing your request. Please try again later.")
            logging.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()