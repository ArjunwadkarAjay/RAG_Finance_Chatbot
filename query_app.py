import streamlit as st
from rag_helper import * # Assuming rag_helper.py contains the necessary functions

# Streamlit UI
def main():
    st.title("Financial RAG Chatbot")
    st.text("Query about financial statements for the company's data based on product, category, sales, etc!")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    query = st.text_input("You:", "")

    if query:
        st.session_state['chat_history'].append({"role": "user", "content": query})

        if not guardrail_input(query):
            response = "This query is not financial-related or is unsafe."
        else:
            results = hybrid_search(query, k=5)
            ranked_results = rerank_results(query, results)
            response = generate_response(query, " ".join(ranked_results))

            if not guardrail_output(response):
                response = "The generated response may be misleading. Please verify with official sources."

        st.session_state['chat_history'].append({"role": "assistant", "content": response})

    # Display chat history
    for message in st.session_state['chat_history']:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Bot:** {message['content']}")

if __name__ == "__main__":
    main()