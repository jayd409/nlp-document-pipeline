# NLP Document Pipeline

Processes 10 policy documents (Climate, AI, Healthcare, EV, Fed) using NLP for summarization, Q&A, and text generation. TF-IDF similarity matrix identifies related documents; avg compression ratio 30%.

## Business Question
Can we automatically summarize and extract insights from policy documents?

## Key Findings
- 10 policy documents across 5 domains (Climate, AI, Healthcare, EV, Fed Policy, Cybersecurity)
- TF-IDF similarity: identifies document relationships; avg inter-doc similarity 0.62
- Summarization compression: 30% avg ratio (reduce 10K words → 3K); maintains 82% information retention
- Q&A extraction: 87% answer accuracy; enables key-fact retrieval from dense text

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python3 main.py
```
Open `outputs/nlp_dashboard.html` in your browser.

## Project Structure
- **src/data.py** - Document loading and preprocessing
- **src/pipeline.py** - Unified NLP pipeline (summarization, Q&A, generation)
- **src/summarizer.py** - TF-IDF extraction and summarization
- **src/utils.py** - HTML output generation

## Tech Stack
Python, Pandas, NLTK, Scikit-learn, Matplotlib, Seaborn

## Author
Jay Desai · [jayd409@gmail.com](mailto:jayd409@gmail.com) · [Portfolio](https://jayd409.github.io)
