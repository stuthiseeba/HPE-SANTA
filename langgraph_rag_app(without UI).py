import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from langgraph.graph import StateGraph
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
import ollama
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel  # ✅ FIXED: Use direct Pydantic import

# ✅ Define the State Schema
class QueryState(BaseModel):
    query: str
    retrieved_docs: list = []
    context: str = ""
    mistral_response: str = ""  # ✅ Added this to avoid KeyError

# ✅ Connect to Qdrant
qdrant = QdrantClient("http://localhost:6333")
print("✅ Qdrant is connected!")

# ✅ Function to read text from a PDF file
def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = "\n".join([page.get_text() for page in doc])
    return text

# 📂 Load text from the uploaded PDF
pdf_path = "Autonomous-Networks-whitepaper"
text = read_pdf(pdf_path)

# ✅ Use RecursiveCharacterTextSplitter for better chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, 
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

docs = text_splitter.create_documents([text])

# ✅ Load HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

# ✅ Store documents in Qdrant
vectorstore = Qdrant.from_documents(
    documents=docs,
    embedding=embeddings,
    location="http://localhost:6333",
    collection_name="resume_data",
)
print("✅ Resume data stored in Qdrant!")

# ✅ Define LangGraph Workflow with Schema
workflow = StateGraph(QueryState)

# Step 1: Retrieve documents
def retrieve_docs(state: QueryState):
    docs_with_scores = vectorstore.similarity_search_with_score(state.query, k=10)

    # Filter documents with scores above 0.75
    relevant_docs = [doc for doc, score in docs_with_scores if score > 0.75]
    
    # If not enough relevant docs, re-run retrieval with broader query
    if len(relevant_docs) < 2:
        print("🔄 Not enough relevant docs. Expanding query...")
        broader_query = f"Find anything related to {state.query}"
        broader_results = vectorstore.similarity_search_with_score(broader_query, k=30)
        relevant_docs += [doc for doc, score in broader_results if score > 0.70]

    return QueryState(query=state.query, retrieved_docs=relevant_docs)

workflow.add_node("retrieve", retrieve_docs)

# Step 2: Format Context for Mistral
def format_context(state: QueryState):
    context = "\n".join([doc.page_content for doc in state.retrieved_docs])
    return QueryState(query=state.query, retrieved_docs=state.retrieved_docs, context=context)

workflow.add_node("format", format_context)

# Step 3: Query Mistral
def query_mistral(state: QueryState):
    response = ollama.chat(model="mistral", messages=[
        {"role": "system", "content": "You are an AI assistant that only provides relevant answers."},
        {"role": "user", "content": f"Here is the extracted information:\n\n{state.context}"},
        {"role": "user", "content": f"{state.query}"}
    ])

    return QueryState(
        query=state.query,
        retrieved_docs=state.retrieved_docs,
        context=state.context,
        mistral_response=response["message"]["content"]
    )

workflow.add_node("query_mistral", query_mistral)

# ✅ Define Execution Flow
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "format")
workflow.add_edge("format", "query_mistral")

# ✅ Compile the Graph
graph = workflow.compile()

# ✅ Run the Workflow with a Query
query = "List the projects completed by this person, including project names and descriptions."

state = graph.invoke(QueryState(query=query))

# ✅ Print Final Answer
print("\n🤖 Mistral's Answer:", state["mistral_response"])  # ✅ FIXED KeyError Issue

