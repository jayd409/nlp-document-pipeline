import re
from collections import Counter

def tokenize(text):
    text = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text)
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'be', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'must', 'can', 'it', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'we', 'they'
    }
    return [t for t in tokens if t not in stopwords and len(t) > 2]

def split_sentences(text):
    sents = re.split(r'[.!?]+', text)
    return [s.strip() for s in sents if s.strip()]

def generate_continuation(prompt, corpus):
    """
    Template-based text generation. Given a prompt, find sentences from corpus
    that continue the thought thematically.
    """
    prompt_tokens = set(tokenize(prompt))

    all_sents = []
    for doc in corpus:
        all_sents.extend(split_sentences(doc))

    continuations = []
    for sent in all_sents:
        if len(sent.split()) < 8:
            continue

        sent_tokens = set(tokenize(sent))
        overlap = len(prompt_tokens & sent_tokens)

        if overlap > 0 or len(continuations) < 2:
            continuations.append((sent, overlap))

    continuations.sort(key=lambda x: x[1], reverse=True)

    if not continuations:
        result = " ".join(prompt.split()[:5]) + " [completion unavailable]"
    else:
        first = continuations[0][0]
        first_words = first.split()[:6]
        result = prompt + " " + " ".join(first_words)

    return result
