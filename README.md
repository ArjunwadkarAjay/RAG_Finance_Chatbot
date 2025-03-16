## Project Info.

Use virtual env and use requirements.txt to set the environment.

Install Ollama locally & install / pull llama3 model to serve it

Use ```streamlit run query_app.py``` to start the required streamlit ui code

### Explanation of the Logic in the RAG Chatbot

This **Retrieval-Augmented Generation (RAG) chatbot** is designed to answer **financial queries** by retrieving relevant financial statements and using **Llama** via Ollama to generate responses. Here’s a breakdown of its components and logic:

#### **1. Data Preparation**
- The chatbot **loads a preprocessed CSV file** containing financial statements.
- It **cleans the text** by removing special characters (`[^a-zA-Z0-9.,\s]`).
- The **cleaned text is embedded** using `all-MiniLM-L6-v2` (a SentenceTransformer model).

#### **2. Retrieval Mechanisms**
The chatbot employs **two retrieval methods**:
- **Dense Retrieval (FAISS)**
  - Converts the query into an embedding.
  - Uses **FAISS (Facebook AI Similarity Search)** to find the most similar documents in vector space.
- **Sparse Retrieval (BM25)**
  - Uses **BM25** (Bag-of-Words retrieval) to score documents based on token matches.

#### **3. Hybrid Search**
- Combines **dense FAISS** and **sparse BM25** results.
- Merges the top results from both, ensuring **diversity in retrieval**.

#### **4. Re-Ranking**
- Uses a **cross-encoder (`ms-marco-MiniLM-L-6-v2`)** to re-rank retrieved documents.
- Ensures the most **contextually relevant** documents appear first.

#### **5. Response Generation**
- Constructs a **prompt** using the retrieved context and user query.
- Calls **Llama** via Ollama to generate an answer.

#### **6. Guardrails for Safety**
- **Input Filtering:** Uses Llama to check if the query is **financial-related and safe**.
- **Output Filtering:** Ensures the response does not contain **hallucinations or financial inaccuracies**.

---

### **Summary**
This RAG chatbot **retrieves relevant financial data**, refines search results, and then generates accurate responses using **Llama**. This hybrid approach **reduces hallucinations** and **ensures factual accuracy**, making it more reliable than standard LLMs.

Let me know if you need further refinements! 🚀