import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["MLFLOW_ENABLE_ARTIFACTS_DESTINATION_OVERRIDE"] = "false"

import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langgraph.graph import StateGraph
from pydantic import BaseModel
import ollama
import mlflow
import hashlib
import time



# --- Streamlit App Config ---
st.set_page_config(page_title="NotebookLM-style Q&A", layout="wide")
st.title("Local RAG App")

model_choice = st.selectbox("Select a model", ["mistral","llama3"])
# --- State Class ---
class QueryState(BaseModel):
    query: str
    retrieved_docs: list = []
    context: str = ""
    model_response: str = ""

# --- Upload PDF ---
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # --- Read PDF Text ---
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])

    # --- Split and Deduplicate Chunks ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([text])
    seen = set()
    unique_docs = []
    for doc in docs:
        content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
        if content_hash not in seen:
            seen.add(content_hash)
            unique_docs.append(doc)

    # --- Embeddings & Vector DB ---
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    qdrant = QdrantClient("http://localhost:6333")

    vectorstore = Qdrant.from_documents(
        documents=unique_docs,
        embedding=embeddings,
        location="http://localhost:6333",
        collection_name="document2_data",
        force_recreate=True  # Reset if embedding dim mismatches
    )

    st.success("✅ PDF content embedded and stored in Qdrant.")

    # --- LangGraph Nodes ---
    workflow = StateGraph(QueryState)

    def retrieve_docs(state: QueryState):
        docs_with_scores = vectorstore.similarity_search_with_score(state.query, k=5)
        relevant_docs = [doc for doc, score in docs_with_scores]
        return QueryState(query=state.query, retrieved_docs=relevant_docs)

    def format_context(state: QueryState):
        context = "\n---\n".join([doc.page_content for doc in state.retrieved_docs])
        return QueryState(query=state.query, retrieved_docs=state.retrieved_docs, context=context)

    def query_mistral(state: QueryState):
        prompt = [
         
            {"role": "system", "content": "You are a helpful assistant. Answer using ONLY the provided context. If the answer is not in the context, say you don't know."},
            {"role": "user", "content": f"""Answer the following question using the context below.
            
            Context:
            {state.context}
            
            Question:
            {state.query}
            """}

        ]
        
        
        response = ollama.chat(model=model_choice, messages=prompt)
        return QueryState(
            query=state.query,
            retrieved_docs=state.retrieved_docs,
            context=state.context,
            model_response=response["message"]["content"]
        )

    workflow.add_node("retrieve", retrieve_docs)
    workflow.add_node("format", format_context)
    workflow.add_node("query_mistral", query_mistral)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "format")
    workflow.add_edge("format", "query_mistral")

    graph = workflow.compile()

    # --- User Query Interface ---
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # --- Chat-style UI ---
    st.subheader("🧠 Ask anything from the document")
    
    if prompt := st.chat_input("Type your question..."):
        with st.spinner("Thinking..."):
            # Process the question
            result_state = graph.invoke(QueryState(query=prompt))
            
            # Save to history
            st.session_state.chat_history.append({
                "query": prompt,
                "response": result_state["model_response"]
            })
    
            # Log to MLflow
            with mlflow.start_run():
                mlflow.set_tag("app", "Local RAG app")
                mlflow.log_param("query", prompt)
                mlflow.log_param("model", model_choice)
                mlflow.log_metric("retrieved_doc_count", len(result_state["retrieved_docs"]))
                # Save retrieved context as a local file and log it as artifact
                

                mlflow.log_param("timestamp", int(time.time()))  # for run ordering
                mlflow.log_param("prompt_template_version", "v1")  # update if you modify prompt style
            
                context_length = len(result_state["context"])
                response_length = len(result_state["model_response"])
            
                mlflow.log_metric("retrieved_doc_count", len(result_state["retrieved_docs"]))
                mlflow.log_metric("context_length_chars", context_length)
                mlflow.log_metric("response_length_chars", response_length)

            
    
    # Display conversation
    for i, turn in enumerate(st.session_state.chat_history):
        st.chat_message("user").write(turn["query"])
        st.chat_message("assistant").write(turn["response"])
    
        