import streamlit as st
import json
from paper_processor import PaperProcessor
from api_client import ArxivAPIClient
from keyword_extractor import KeywordExtractor
from cache_manager import CacheManager

st.set_page_config(page_title="Research Paper Discovery System", layout="wide")

def main():
    st.title("Research Paper Discovery System")
    
    # Initialize components
    cache_manager = CacheManager()
    keyword_extractor = KeywordExtractor()
    arxiv_client = ArxivAPIClient()
    paper_processor = PaperProcessor(keyword_extractor, arxiv_client, cache_manager)
    
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
            with st.spinner("Processing your request..."):
                results = paper_processor.process_query(research_question, criteria)
                
                # Display results
                st.subheader("Found Papers")
                for paper in results['papers']:
                    with st.expander(f"{paper['title']}"):
                        st.write(f"**Authors:** {', '.join(paper['authors'])}")
                        st.write(f"**Published:** {paper['published']}")
                        st.write(f"**Abstract:** {paper['abstract']}")
                        st.write(f"**arXiv ID:** {paper['arxiv_id']}")
                        st.write(f"**URL:** {paper['url']}")
                
                # Display extracted keywords
                st.sidebar.subheader("Extracted Keywords")
                st.sidebar.write(results['keywords'])
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
