import faiss
import ollama
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import pandas as pd

# Load embedding model (Open-source)
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Load financial data (preprocessed CSV)
data = pd.read_csv("./dataset/financial_statements.csv")

# Function to clean text and form structured sentences
def format_financial_data(row):
    return (f"In {row['Month Name']} {row['Year']}, the {row['Segment']} sector in {row['Country']} sold "
            f"{row['Units Sold']} units of {row['Product']} at a sale price of {row['Sale Price']} per unit. "
            f"The total sales amounted to {row['Sales']}, with a gross profit of {row['Profit']}.")

# Apply the function to each row
data['clean_text'] = data.apply(format_financial_data, axis=1)

# Convert financial statements into embeddings
texts = data['clean_text'].tolist()

# Convert financial statements into embeddings
texts = data['clean_text'].tolist()
embeddings = embed_model.encode(texts, convert_to_tensor=True)

# Initialize FAISS vector DB
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings.cpu().numpy())

def search_faiss(query, k=5):
    """Performs dense retrieval using FAISS."""
    query_embedding = embed_model.encode([query], convert_to_tensor=True)
    D, I = index.search(query_embedding.cpu().numpy(), k)
    return [texts[i] for i in I[0]]

def bm25_search(query, corpus, k=5):
    """Performs sparse retrieval using BM25."""
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.split())
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [corpus[i] for i in ranked_indices[:k]]

def hybrid_search(query, k=5):
    """Combines FAISS (dense) and BM25 (sparse) retrieval."""
    dense_results = search_faiss(query, k)
    sparse_results = bm25_search(query, texts, k)
    combined_results = list(set(dense_results + sparse_results))[:k]
    return combined_results

def rerank_results(query, retrieved_docs):
    """Re-ranks retrieved documents using cross-encoder."""
    scores = cross_encoder.predict([(query, doc) for doc in retrieved_docs])
    ranked_docs = [doc for _, doc in sorted(zip(scores, retrieved_docs), reverse=True)]
    return ranked_docs


def generate_response(query, context):
    """Generates response using Llama via Ollama with retrieved context."""
    prompt = f"""Given the following financial statements, answer the query:
    Context: {context}
    Query: {query}
    Answer:"""
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def guardrail_input(query):
    """Filters inappropriate queries using Llama via Ollama."""
    guardrail_prompt = f"""Is the following query financial-related and safe?
    Query: {query}
    Response (Yes/No):"""
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": guardrail_prompt}])
    return "yes" in response["message"]["content"].strip().lower()

def guardrail_output(response):
    """Filters hallucinated/misleading outputs using Llama via Ollama."""
    check_prompt = f"""Does the following response contain financial inaccuracies or hallucinations?
    Response: {response}
    Answer (Yes/No):"""
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": check_prompt}])
    return "no" in response["message"]["content"].strip().lower()

