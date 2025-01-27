import json
from paper_analyzer import PaperAnalyzer

def debug_paper_analysis():
    # Sample papers data for testing
    sample_papers = [
        {
            'title': 'Attention Is All You Need',
            'arxiv_id': '1706.03762',  # Transformer paper
            'abstract': 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...'
        },
        {
            'title': 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
            'arxiv_id': '1810.04805',  # BERT paper
            'abstract': 'We introduce a new language representation model called BERT...'
        }
    ]

    research_question = "What are the recent advances in transformer architectures and their impact on NLP?"

    print("🚀 Starting paper analysis debug...")
    print(f"Research Question: {research_question}")
    print(f"Testing with {len(sample_papers)} sample papers\n")

    # Initialize analyzer
    analyzer = PaperAnalyzer()

    # Run analysis
    try:
        print("📥 Downloading and analyzing papers...")
        results = analyzer.analyze_papers(sample_papers, research_question)

        # Print results in a readable format
        print("\n📊 Analysis Results:")
        for i, result in enumerate(results, 1):
            print(f"\nPaper {i}: {result['title']}")
            print("=" * 50)
            if 'error' in result['analysis']:
                print(f"❌ Error: {result['analysis']['error']}")
            else:
                print("📝 Summary:")
                print(result['analysis'].get('summary', 'No summary available'))
                print("\n🎯 Relevant Points:")
                for point in result['analysis'].get('relevant_points', []):
                    print(f"• {point}")
                if result['analysis'].get('limitations'):
                    print("\n⚠️ Limitations:")
                    for limitation in result['analysis']['limitations']:
                        print(f"• {limitation}")
            print("-" * 50)

    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")

if __name__ == "__main__":
    debug_paper_analysis()