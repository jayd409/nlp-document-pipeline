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

def cosine_sim(vec1, vec2):
    """Cosine similarity between two frequency vectors."""
    common = set(vec1.keys()) & set(vec2.keys())
    if not common:
        return 0.0

    dot = sum(vec1[k] * vec2[k] for k in common)
    mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v**2 for v in vec2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)

def split_sentences(text):
    sents = re.split(r'[.!?]+', text)
    return [s.strip() for s in sents if s.strip()]

def answer_question(question, docs):
    """
    Q&A retrieval using cosine similarity of word-frequency vectors.
    Returns top-3 most relevant sentences with scores.
    """
    q_tokens = Counter(tokenize(question))

    results = []
    for doc_text in docs:
        sents = split_sentences(doc_text)
        for sent in sents:
            if len(sent.split()) < 5:
                continue

            sent_tokens = Counter(tokenize(sent))
            score = cosine_sim(q_tokens, sent_tokens)

            if score > 0:
                results.append((sent, score))

    results.sort(key=lambda x: x[1], reverse=True)
    top_3 = results[:3]

    return top_3
