import re
from collections import Counter
import math

def tokenize(text):
    text = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text)
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'be', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'must', 'can', 'it', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'we', 'they', 'what', 'which', 'who', 'when',
        'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
        'most', 'other', 'some', 'any', 'many', 'much', 'such', 'so', 'than',
        'not', 'no', 'nor', 'only', 'just', 'even', 'also', 'back', 'up', 'out'
    }
    return [t for t in tokens if t not in stopwords and len(t) > 2]

def calc_tfidf(docs):
    doc_tokens = [tokenize(doc) for doc in docs]
    idf = {}
    for tokens in doc_tokens:
        for tok in set(tokens):
            idf[tok] = idf.get(tok, 0) + 1

    for tok in idf:
        idf[tok] = math.log(len(docs) / idf[tok])

    return doc_tokens, idf

def score_sentences(text, idf):
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return sentences, []

    scores = []
    for sent in sentences:
        tokens = tokenize(sent)
        if not tokens:
            scores.append(0)
            continue
        score = sum(idf.get(tok, 0) for tok in tokens) / len(tokens)
        scores.append(score)

    return sentences, scores

def summarize(text, ratio=0.3):
    """
    Extractive summarization using TF-IDF sentence scoring.
    Ratio controls compression (0.3 = keep 30% of sentences).
    """
    sentences, scores = score_sentences(text, {})

    if not sentences:
        return text, 1.0

    if len(scores) == 1:
        return text, 1.0

    num_to_keep = max(1, int(len(sentences) * ratio))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_to_keep]
    top_indices.sort()

    summary = ' '.join(sentences[i] for i in top_indices)

    orig_words = len(text.split())
    summary_words = len(summary.split())
    compression = summary_words / orig_words if orig_words > 0 else 1.0

    return summary, compression

def get_top_terms(docs, num=10):
    """Extract top TF-IDF terms across documents."""
    all_tokens = []
    for doc in docs:
        tokens = tokenize(doc)
        all_tokens.extend(tokens)

    freq = Counter(all_tokens)
    idf_vals = {}

    for tok in freq:
        doc_count = sum(1 for doc in docs if tok in tokenize(doc))
        idf_vals[tok] = math.log(len(docs) / (1 + doc_count))

    tfidf_scores = {tok: freq[tok] * idf_vals[tok] for tok in freq}
    top = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:num]

    return top
