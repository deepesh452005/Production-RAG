from sentence_transformers import CrossEncoder

# Load the reranker model (this downloads it the first time, ~90MB)
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# A sample query relevant to AI history, and a few sample chunks
query = "When was the term artificial intelligence first coined?"

chunks = [
    "The term 'artificial intelligence' was coined in 1956 at the Dartmouth Conference by John McCarthy.",
    "Machine learning is a subset of AI that focuses on algorithms learning from data.",
    "The Dartmouth Conference in 1956 is widely considered the founding event of AI as a field.",
    "Modern AI systems use neural networks with billions of parameters for tasks like language generation.",
]

# The reranker needs (query, chunk) pairs
pairs = [(query, chunk) for chunk in chunks]

# Get relevance scores
scores = model.predict(pairs)

# Sort chunks by score, highest first
ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

print("\nRanked results (most relevant first):\n")
for chunk, score in ranked:
    print(f"Score: {score:.4f} | {chunk}")