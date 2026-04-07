"""
NLP Document Pipeline: Summarization, Q&A, and Text Generation

This pipeline demonstrates extractive NLP techniques using TF-IDF scoring
and cosine similarity for multi-task document understanding.

To use real Lamini-Flan-T5:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
    tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")

    def summarize_real(text):
        inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=150, num_beams=4)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

The model checkpoint is optimized for multi-task fine-tuning and supports
summarization, question answering, and translation without task-specific retraining.
"""

import time
from . import summarizer, qa_engine, text_generator
from .data import load_docs

class NLPPipeline:
    def __init__(self):
        self.docs = load_docs()
        self.metrics = {
            'total_docs': len(self.docs),
            'compressions': [],
            'retrieval_times': [],
            'throughput_estimate': None
        }

    def run_summarization(self, ratio=0.3, num_samples=3):
        """Run extractive summarization on sample documents."""
        print("\n=== SUMMARIZATION ===")
        results = []

        sample_docs = self.docs[:num_samples]
        for doc in sample_docs:
            text = doc['text']
            summary, compression = summarizer.summarize(text, ratio=ratio)

            orig_words = len(text.split())
            summary_words = len(summary.split())

            self.metrics['compressions'].append(compression)

            results.append({
                'doc_id': doc['id'],
                'title': doc['title'],
                'original_words': orig_words,
                'summary_words': summary_words,
                'compression_ratio': compression,
                'summary': summary[:200] + '...'
            })

            print(f"\n{doc['title']}")
            print(f"  Original: {orig_words} words → Summary: {summary_words} words")
            print(f"  Compression: {compression:.2%}")

        return results

    def run_qa(self, num_questions=5):
        """Run Q&A retrieval on sample questions."""
        print("\n=== QUESTION ANSWERING ===")
        questions = [
            "What are the main challenges in artificial intelligence?",
            "How has remote work affected productivity?",
            "What renewable energy sources are most promising?",
            "What cybersecurity threats are most critical?",
            "How is machine learning changing healthcare?"
        ]

        questions = questions[:num_questions]
        results = []

        doc_texts = [doc['text'] for doc in self.docs]

        for q in questions:
            start = time.time()
            answers = qa_engine.answer_question(q, doc_texts)
            elapsed = time.time() - start

            self.metrics['retrieval_times'].append(elapsed)

            print(f"\nQ: {q}")
            if answers:
                for i, (ans, score) in enumerate(answers, 1):
                    print(f"  [{i}] (confidence: {score:.2%}) {ans[:100]}...")
            else:
                print("  [No relevant answers found]")

            results.append({
                'question': q,
                'answers': answers,
                'retrieval_time': elapsed
            })

        return results

    def run_text_generation(self, num_samples=3):
        """Run template-based text generation."""
        print("\n=== TEXT GENERATION ===")
        prompts = [
            "The future of artificial intelligence",
            "Climate change and renewable energy",
            "Remote work is transforming"
        ]

        prompts = prompts[:num_samples]
        results = []
        doc_texts = [doc['text'] for doc in self.docs]

        for prompt in prompts:
            continuation = text_generator.generate_continuation(prompt, doc_texts)
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {continuation}")
            results.append({'prompt': prompt, 'continuation': continuation})

        return results

    def compute_metrics(self):
        """Compute pipeline-level metrics."""
        avg_compression = sum(self.metrics['compressions']) / len(self.metrics['compressions']) if self.metrics['compressions'] else 1.0
        avg_retrieval = sum(self.metrics['retrieval_times']) / len(self.metrics['retrieval_times']) if self.metrics['retrieval_times'] else 0.0

        total_docs = self.metrics['total_docs']
        avg_doc_words = 250
        processing_speed = avg_doc_words / (avg_retrieval + 0.1)

        self.metrics['avg_compression'] = avg_compression
        self.metrics['avg_retrieval_time'] = avg_retrieval
        self.metrics['throughput_estimate'] = processing_speed

        print("\n=== PIPELINE METRICS ===")
        print(f"Total Documents: {total_docs}")
        print(f"Avg Compression Ratio: {avg_compression:.2%}")
        print(f"Avg Retrieval Time: {avg_retrieval*1000:.2f}ms")
        print(f"Throughput Est: {processing_speed:.0f} words/sec")

        return self.metrics

    def run_all(self):
        """Execute full pipeline."""
        summary_results = self.run_summarization()
        qa_results = self.run_qa()
        gen_results = self.run_text_generation()
        metrics = self.compute_metrics()

        return {
            'summarization': summary_results,
            'qa': qa_results,
            'generation': gen_results,
            'metrics': metrics
        }
