import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import fitz
import time
import hashlib
import mlflow
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langgraph.graph import StateGraph
from pydantic import BaseModel
import ollama
import re
import json

st.set_page_config(page_title="LLM Judge Battle", layout="wide")
st.title("🤖 LLM Battle | Judged by Your Choice of LLM")

# ---- LLM Selection UI ----
st.sidebar.header("Choose LLMs")
available_llms = ["mistral", "llama3", "zephyr", "tinyllama"]
llm1 = st.sidebar.selectbox("Answer LLM 1", available_llms, index=0)
llm2 = st.sidebar.selectbox("Answer LLM 2", available_llms, index=1)
judge_llm = st.sidebar.selectbox("Judge LLM", available_llms, index=2)

class QueryState(BaseModel):
    query: str
    retrieved_docs: list = []
    context: str = ""
    llm1_response: str = ""
    llm2_response: str = ""
    judge_evaluation: str = ""
    llm1_time: float = 0.0
    llm2_time: float = 0.0
    score_llm1: int = 0
    score_llm2: int = 0

def extract_json_from_text(text):
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            return None
    return None

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    unique_docs = []
    seen = set()
    for doc in docs:
        h = hashlib.md5(doc.page_content.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_docs.append(doc)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    qdrant = QdrantClient("http://localhost:6333")
    vectorstore = Qdrant.from_documents(
        documents=unique_docs,
        embedding=embeddings,
        location="http://localhost:6333",
        collection_name="doc_data",
        force_recreate=True
    )
    st.success("✅ Document embedded!")

    workflow = StateGraph(QueryState)

    def retrieve_docs(state: QueryState):
        docs = vectorstore.similarity_search_with_score(state.query, k=5)
        return QueryState(query=state.query, retrieved_docs=[d for d, _ in docs])

    def format_context(state: QueryState):
        context = "\n---\n".join([doc.page_content for doc in state.retrieved_docs])
        new_state = state.copy()
        new_state.context = context
        return new_state

    def query_model(state: QueryState, model: str):
        prompt = [
            {"role": "system", "content": "Answer ONLY using the context below."},
            {"role": "user", "content": f"""Context:
{state.context}

Question:
{state.query}"""}
        ]
        start = time.time()
        response = ollama.chat(model=model, messages=prompt)
        duration = time.time() - start
        return response["message"]["content"], duration

    def query_both_models(state: QueryState):
        llm1_answer, llm1_time = query_model(state, llm1)
        llm2_answer, llm2_time = query_model(state, llm2)
        return state.copy(update={
            "llm1_response": llm1_answer,
            "llm2_response": llm2_answer,
            "llm1_time": llm1_time,
            "llm2_time": llm2_time
        })

    def evaluate_with_judge_llm(state: QueryState):
        judge_prompt = [
            {"role": "system", "content": "You are a strict AI judge evaluating two answers to the same question, using ONLY the provided document context. Your goal is to determine which answer best responds to the user's query by staying factually accurate and relevant to the context. Do NOT rely on general knowledge. Use only the document content."},
            {"role": "user", "content": f"""---
📚 Context (Extracted from Document):
{state.context[:3000]}

❓ Query:
{state.query}

📝 Answer 1 (from {llm1}):
{state.llm1_response}

📝 Answer 2 (from {llm2}):
{state.llm2_response}

Time 1 (from {llm1}):
{state.llm1_time:.2f} seconds

Time 2 (from {llm2}):
{state.llm2_time:.2f} seconds

Response Length for llm1: {len(state.llm1_response)} characters  
Response Length for llm2: {len(state.llm2_response)} characters

---

🎯 Evaluation Criteria (score each 0-10):
- **Accuracy**: Does the answer contain factually correct info from the context?
- **Completeness**: Does the answer fully answer the query based on the context?
- **Consistency**: Is the reasoning logically coherent throughout?
- **Speed**: Was it generated quickly? Shorter runtime = higher score.
- **Response Length**: Is the length appropriate? Too long/short = lower score.
- **Relevance to Context**: Does it only use info from the given context?
- **Non-bias**: Is the answer fair and objective?
- **Clarity/Precision**: Is it clear, unambiguous, and **precise**?
- **Spelling/Grammar**: Are there any spelling, grammatical, or typographical errors? (10 = flawless, 0 = many errors)
- **Compliance**: Does it follow ethical, safe, and responsible guidelines?

---

🚫 Evaluation Rules:
- Penalize hallucinated content or generic answers not grounded in context.
- If one answer gives specific info like from the document, reward that.
- DO NOT give 10s to answers with hallucinated content.
- If an answer **mentions things not in the document**, penalize.

---

Return the following JSON format **ONLY** :
{{
  "Answer 1": {{
    "Accuracy": <score>,
    "Completeness": <score>,
    "Consistency": <score>,
    "Speed": <score>,
    "Response Length": <score>,
    "Relevance to Context": <score>,
    "Non-bias": <score>,
    "Clarity/Precision": <score>,
    "Spelling/Grammar": <score>,
    "Compliance": <score>
  }},
  "Answer 2": {{
    "Accuracy": <score>,
    "Completeness": <score>,
    "Consistency": <score>,
    "Speed": <score>,
    "Response Length": <score>,
    "Relevance to Context": <score>,
    "Non-bias": <score>,
    "Clarity/Precision": <score>,
    "Spelling/Grammar": <score>,
    "Compliance": <score>
  }},
  "Winner": "Answer 1" or "Answer 2",
  "Explanation": "Clearly explain why the winner was chosen based on the scores and context."
}}
"""}
        ]

        evaluation = ollama.chat(model=judge_llm, messages=judge_prompt)["message"]["content"]

        eval_json = extract_json_from_text(evaluation)
        if eval_json:
            llm1_scores = eval_json.get("Answer 1", {})
            llm2_scores = eval_json.get("Answer 2", {})
            winner = eval_json.get("Winner", "")
            explanation = eval_json.get("Explanation", "")
            score_llm1 = sum(float(v) for v in llm1_scores.values() if isinstance(v, (int, float, float)))
            score_llm2 = sum(float(v) for v in llm2_scores.values() if isinstance(v, (int, float, float)))
        else:
            llm1_scores = {}
            llm2_scores = {}
            winner = ""
            explanation = ""
            score_llm1 = 0
            score_llm2 = 0

        return state.copy(update={
            "judge_evaluation": evaluation,
            "score_llm1": score_llm1,
            "score_llm2": score_llm2,
            "llm1_scores": llm1_scores,
            "llm2_scores": llm2_scores,
            "winner": winner,
            "explanation": explanation
        })

    # Workflow setup
    workflow.add_node("retrieve", retrieve_docs)
    workflow.add_node("format", format_context)
    workflow.add_node("query_models", query_both_models)
    workflow.add_node("judge", evaluate_with_judge_llm)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "format")
    workflow.add_edge("format", "query_models")
    workflow.add_edge("query_models", "judge")

    graph = workflow.compile()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.subheader("🧠 Ask something from the document")

    if prompt := st.chat_input("Your question..."):
        with st.spinner("Evaluating both models and judging..."):
            result = graph.invoke(QueryState(query=prompt))

            st.session_state.chat_history.append({
                "query": prompt,
                "llm1": result["llm1_response"],
                "llm2": result["llm2_response"],
                "judge": result["judge_evaluation"],
                "score_llm1": result["score_llm1"],
                "score_llm2": result["score_llm2"],
            })

            # Log to MLflow
            with mlflow.start_run():
                mlflow.set_tag("comparison", f"{llm1}_vs_{llm2}_judged_by_{judge_llm}")
                mlflow.log_param("query", prompt)
                mlflow.log_param("context_length", len(result["context"]))
                mlflow.log_param(f"{llm1}_time", result["llm1_time"])
                mlflow.log_param(f"{llm2}_time", result["llm2_time"])
                mlflow.log_param(f"{llm1}_response_length", len(result["llm1_response"]))
                mlflow.log_param(f"{llm2}_response_length", len(result["llm2_response"]))
                mlflow.log_param("llm1",llm1)
                mlflow.log_param("llm2",llm2)
                mlflow.log_param("judge_llm",judge_llm)

                # Generic metric logging for all answers/metrics in judge_evaluation
                try:
                    scores = extract_json_from_text(result["judge_evaluation"])
                    if isinstance(scores, dict):
                        for answer_key, metrics in scores.items():
                            if isinstance(metrics, dict):
                                for metric, score in metrics.items():
                                    try:
                                        mlflow.log_metric(f"{answer_key}_{metric}", float(score))
                                    except Exception:
                                        pass  # skip non-numeric
                except Exception:
                    pass

                # Also log winner/explanation as tags if present
                if "winner" in result:
                    mlflow.set_tag("winner", result["winner"])
                if "explanation" in result:
                    mlflow.set_tag("explanation", result["explanation"])

    for turn in st.session_state.chat_history:
        st.chat_message("user").write(turn["query"])

        col1, col2 = st.columns(2)
        with col1:
            st.chat_message("assistant").markdown(f"### 🤖 {llm1}\n" + turn["llm1"])
        with col2:
            st.chat_message("assistant").markdown(f"### 🤖 {llm2}\n" + turn["llm2"])

        with st.expander("🧑‍⚖️ Judge's Full Evaluation"):
            st.markdown(turn["judge"])
