# app_crew.py - Complete Enhanced Multi-Agent Autonomous Network Copilot
import os, re, time, tempfile
from datetime import datetime
from io import BytesIO
from collections import Counter
from typing import Dict, List, Tuple  # FIXED: Added Tuple import
import streamlit as st
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyttsx3

# ---------- LlamaIndex / Ollama / Qdrant ----------
# FORCE LOCAL ONLY - NO OPENAI
import os
# Disable OpenAI completely
os.environ["OPENAI_API_KEY"] = "dummy"  # Prevent OpenAI initialization
os.environ["LLAMA_INDEX_DISABLE_OPENAI"] = "1"

try:
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
    from llama_index.core.prompts import PromptTemplate
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    
    # FORCE LOCAL SETTINGS IMMEDIATELY
    Settings.llm = None  # Prevent default OpenAI
    Settings.embed_model = None  # Prevent default OpenAI
    
except Exception:
    try:
        from llama_index import SimpleDirectoryReader, VectorStoreIndex, StorageContext
        from llama_index.llms.ollama import Ollama
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.vector_stores import QdrantVectorStore
        from llama_index import Prompt as PromptTemplate
    except Exception:
        SimpleDirectoryReader = None
        VectorStoreIndex = None
        StorageContext = None
        Ollama = None
        OllamaEmbedding = None
        QdrantVectorStore = None
        PromptTemplate = None

try:
    import qdrant_client
except Exception:
    qdrant_client = None

# ==================================================
#                 STREAMLIT UI SETUP
# ==================================================
st.set_page_config(
    page_title="🤖 Multi-Agent Autonomous Network Copilot",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { text-align: center; color: #2E86AB; margin-bottom: .25rem; }
    .caption { text-align: center; color: #9aa0a6; margin-bottom: 1rem; }
    .chat-container { background: #0f1116; color: #e8eaed; border: 1px solid #2b2f36; padding: 1rem; border-radius: 12px; }
    .status-box { border-radius: 10px; padding: .75rem 1rem; margin:.25rem 0 .5rem; }
    .success-box { background:#112d17; border:1px solid #1c6e3c; color:#d7ffdf; }
    .info-box    { background:#132236; border:1px solid #2a5b8a; color:#d6ecff; }
    .warn-box    { background:#332d13; border:1px solid #8a6f2a; color:#fff0c2; }
    .debug-box   { background:#2d1b33; border:1px solid #6e3c8a; color:#e6d6ff; }
</style>
""", unsafe_allow_html=True)
st.markdown('<h1 class="main-header">🤖 Multi-Agent Autonomous Network Copilot (Debug Version)</h1>', unsafe_allow_html=True)
st.markdown('<div class="caption">Local Ollama • Qdrant RAG • Diagram • Text/Audio Transcript</div>', unsafe_allow_html=True)

# ==================================================
#                   GLOBAL PROMPTS - FIXED
# ==================================================
PROMPT_STRICT = """You are an expert assistant for autonomous network documentation. You must provide detailed, technical answers based ONLY on the provided context.

CRITICAL INSTRUCTIONS:
1) Answer ONLY from the provided context below. Use the exact information, terms, and concepts from the context.
2) If the context contains numbered levels, sections, or lists, reproduce them accurately with proper formatting.
3) Be comprehensive and detailed - include all relevant information from the context.
4) Use technical terminology from the context and explain concepts thoroughly.
5) If insufficient information, say: "Based on the provided documents, I don't have sufficient information to answer this question fully."
6) Structure your answer clearly with proper paragraphs and formatting.
7) Include specific details, numbers, and examples from the context when available.

Context:
----------------
{context_str}
----------------

Question: {query_str}

Provide a detailed, comprehensive answer using the context above:
"""

# Fix for PromptTemplate handling
if PromptTemplate:
    try:
        PromptObj = PromptTemplate(PROMPT_STRICT)
    except:
        PromptObj = PROMPT_STRICT
else:
    PromptObj = PROMPT_STRICT

# ==================================================
#                SESSION STATE DEFAULTS
# ==================================================
def _init_state():
    ss = st.session_state
    ss.setdefault("agents", {
        "Supervisor": {"role": "Routes queries & plans"},
        "Retriever": {"role": "Retrieves context from vector DB"},
        "Responder": {"role": "Answers strictly from retrieved context"},
    })
    ss.setdefault("edges", {("Supervisor", "Retriever"), ("Retriever", "Responder"), ("Responder", "Supervisor")})
    ss.setdefault("docs_loaded", False)
    ss.setdefault("docs_dir", None)
    ss.setdefault("index", None)
    ss.setdefault("used_qdrant", False)
    ss.setdefault("chat", [])  # [{from,to,question,answer,context,timestamp}]
    ss.setdefault("model", "llama3")
    ss.setdefault("k", 5)  # Increased from 3 to 5 for better context
    ss.setdefault("strict", True)
    ss.setdefault("show_visual", False)
    ss.setdefault("show_text_download", False)
    ss.setdefault("show_audio_download", False)
    ss.setdefault("session_start", time.time())
    ss.setdefault("show_dev_console", False)
    ss.setdefault("debug_mode", True)  # NEW: Enable debug by default
    ss.setdefault("last_context", "")  # NEW: Store last retrieved context
_init_state()

# ==================================================
#           DIAGRAM / AUDIO HELPERS (IMPROVED)
# ==================================================
def extract_network_entities_from_context(answer_text: str, context_text: str) -> Dict[str, List[str]]:
    """Enhanced entity extraction with better patterns"""
    text = f"{answer_text} {context_text}".lower()
    ent = {'nodes':[], 'connections':[], 'layers':[], 'protocols':[], 'components':[], 'levels':[]}
    
    # Enhanced patterns for autonomous networks
    node_patterns = [
        r'\b(?:router|switch|server|controller|gateway|firewall|load balancer|proxy|hub|bridge|node|device|endpoint|host|client|workstation|orchestrator|manager)\b',
        r'\b(?:core|edge|access|distribution|aggregation|spine|leaf|autonomous|cognitive|intent|policy)\s+(?:router|switch|node|system|engine)',
    ]
    
    # Autonomous network level patterns
    level_patterns = [
        r'\b(?:level|l)\s*[0-6]\b',
        r'\b(?:basic|managed|network|business|intent|fully)\s+(?:autonomous|automation|management)\b',
        r'\bautonomous\s+(?:level|network)\s*[0-6]?\b'
    ]
    
    connection_patterns = [
        r'(\w+)\s+(?:connects to|connected to|linked to|communicates with|sends to|receives from)\s+(\w+)',
        r'(\w+)\s+(?:→|->|flows to|routes to|manages)\s+(\w+)',
        r'between\s+(\w+)\s+and\s+(\w+)'
    ]
    
    layer_patterns = [
        r'\b(?:physical|data link|network|transport|session|presentation|application)\s+layer\b', 
        r'\bl[0-7]\b',
        r'\b(?:infrastructure|orchestration|business|intent|policy)\s+(?:layer|plane)\b'
    ]
    
    protocol_patterns = [
        r'\b(?:tcp|udp|ip|http|https|ftp|dns|dhcp|snmp|bgp|ospf|rip|stp|vlan|mpls|vpn|ssl|tls|netconf|restconf|grpc)\b', 
        r'\b(?:ethernet|wifi|802\.11|bluetooth|zigbee|sdn|nfv)\b'
    ]
    
    component_patterns = [
        r'\b(?:database|api|service|application|interface|module|engine|processor|memory|storage|analytics|ml|ai)\b', 
        r'\b(?:microservice|container|virtualization|orchestration|automation|telemetry|monitoring)\b'
    ]

    for p in node_patterns: ent['nodes'] += [m.title() for m in re.findall(p, text)]
    for p in level_patterns: ent['levels'] += [m.upper() for m in re.findall(p, text)]
    for p in connection_patterns: ent['connections'] += [(m[0].title(), m[1].title()) for m in re.findall(p, text) if len(m)==2]
    for p in layer_patterns: ent['layers'] += [m.title() for m in re.findall(p, text)]
    for p in protocol_patterns: ent['protocols'] += [m.upper() for m in re.findall(p, text)]
    for p in component_patterns: ent['components'] += [m.title() for m in re.findall(p, text)]
    
    return {k: sorted(set([x for x in v if len(str(x))>1])) for k,v in ent.items()}

def _concept_diagram(answer: str, question: str):
    """Improved concept diagram generation"""
    G = nx.Graph()
    # Look for autonomous network specific terms
    autonomous_terms = re.findall(r'\b(?:autonomous|automation|intent|policy|orchestration|cognitive|level|management)\w*\b', 
                                 (answer+" "+question).lower())
    general_terms = re.findall(r'\b\w{4,}\b', (answer+" "+question).lower())
    
    # Prioritize autonomous network terms
    if autonomous_terms:
        words = autonomous_terms + general_terms
    else:
        words = general_terms
        
    top = [w.title() for w,_ in Counter(words).most_common(8)]
    if not top:
        G.add_nodes_from(["Question","Answer","Context"])
        G.add_edges_from([("Question","Answer"), ("Context","Answer")])
        return G, "Conceptual Overview"
    
    G.add_nodes_from(top)
    # Create a more interesting topology
    center = top[0]
    for i, node in enumerate(top[1:], 1):
        G.add_edge(center, node)
        if i > 1 and i < len(top)-1:  # Add some cross-connections
            G.add_edge(top[i], top[i+1])
    
    return G, "Autonomous Network Concepts"

# ==================================================
#        ENHANCED DIAGRAM GENERATION - FIXED
# ==================================================
def extract_network_entities_from_answer(answer_text: str, context_text: str) -> Dict[str, List[str]]:
    """Enhanced entity extraction focused on the generated answer"""
    # Prioritize the answer text, use context as backup
    primary_text = answer_text.lower()
    secondary_text = context_text.lower()
    
    entities = {
        'levels': [],
        'nodes': [], 
        'connections': [],
        'components': [],
        'protocols': [],
        'concepts': []
    }
    
    # Enhanced autonomous network level patterns - prioritize answer content
    level_patterns = [
        r'\b(?:level|l)\s*([0-6])\b',
        r'\b([0-6])\s*(?:autonomous|automation|level)\b',
        r'\b(?:basic|managed|network|business|intent|fully)\s+(?:autonomous|automation)\b',
        r'\b(?:manual|assisted|conditional|high|full)\s+(?:automation|autonomous)\b'
    ]
    
    # Extract levels from answer first
    for pattern in level_patterns:
        matches = re.findall(pattern, primary_text)
        for match in matches:
            if match.isdigit():
                entities['levels'].append(f"Level {match}")
            else:
                entities['levels'].append(match.title())
    
    # If no levels in answer, check context
    if not entities['levels']:
        for pattern in level_patterns:
            matches = re.findall(pattern, secondary_text)
            for match in matches:
                if match.isdigit():
                    entities['levels'].append(f"Level {match}")
                else:
                    entities['levels'].append(match.title())
    
    # Enhanced component extraction focused on autonomous networks
    component_patterns = [
        r'\b(?:controller|orchestrator|manager|engine|analytics|monitoring|telemetry|ai|ml|intent|policy|automation)\b',
        r'\b(?:sdn|nfv|api|interface|dashboard|portal|service|microservice|container)\b',
        r'\b(?:network|infrastructure|management|business|application)\s+(?:layer|plane|domain)\b'
    ]
    
    for pattern in component_patterns:
        entities['components'].extend(re.findall(pattern, primary_text))
    
    # Extract key concepts from answer
    concept_patterns = [
        r'\b(?:autonomous|automation|intent|policy|orchestration|cognitive|intelligent|adaptive|self)\w*\b',
        r'\b(?:zero|closed|open)\s+(?:loop|touch)\b',
        r'\b(?:machine|artificial)\s+(?:learning|intelligence)\b'
    ]
    
    for pattern in concept_patterns:
        entities['concepts'].extend(re.findall(pattern, primary_text))
    
    # Clean and deduplicate
    for key in entities:
        entities[key] = sorted(list(set([item.title() for item in entities[key] if len(str(item)) > 2])))
    
    return entities

def create_dynamic_network_diagram(answer: str, context: str, question: str) -> BytesIO:
    """Enhanced diagram generation based on model's answer"""
    
    # Extract entities with focus on the answer
    entities = extract_network_entities_from_answer(answer, context)
    
    if st.session_state.debug_mode:
        total_entities = sum(len(v) for v in entities.values())
        st.markdown(f'<div class="status-box debug-box">🎨 Diagram entities: {total_entities} total ({entities})</div>', 
                   unsafe_allow_html=True)
    
    # Priority 1: Autonomous Levels (most important for autonomous networks)
    if entities['levels'] and len(entities['levels']) >= 2:
        G = nx.DiGraph()
        levels = entities['levels'][:6]  # Max 6 levels
        
        # Add nodes with hierarchical positioning
        for level in levels:
            G.add_node(level)
        
        # Create hierarchical connections
        for i in range(len(levels)-1):
            G.add_edge(levels[i], levels[i+1])
        
        # Add some components if available
        if entities['components']:
            main_components = entities['components'][:3]
            for comp in main_components:
                G.add_node(comp)
                # Connect components to middle levels
                if len(levels) > 2:
                    G.add_edge(levels[len(levels)//2], comp)
        
        title = "Autonomous Network Levels"
        layout_func = lambda g: nx.hierarchical_layout(g, prog='dot') if hasattr(nx, 'hierarchical_layout') else nx.spring_layout(g, k=2, iterations=50)
        node_color = "lightgreen"
        
    # Priority 2: Components and Concepts
    elif entities['components'] or entities['concepts']:
        G = nx.Graph()
        all_items = (entities['components'] + entities['concepts'])[:8]
        
        if all_items:
            G.add_nodes_from(all_items)
            # Create a hub topology with the first item as hub
            hub = all_items[0]
            for item in all_items[1:]:
                G.add_edge(hub, item)
            
            # Add some cross-connections for complexity
            if len(all_items) > 3:
                for i in range(1, min(4, len(all_items)-1)):
                    G.add_edge(all_items[i], all_items[i+1])
        
        title = "Autonomous Network Components"
        layout_func = nx.spring_layout
        node_color = "lightblue"
        
    # Priority 3: Concept mapping from answer
    else:
        G = nx.Graph()
        
        # Extract key terms from the answer itself
        answer_words = re.findall(r'\b\w{5,}\b', answer.lower())
        important_words = [w for w in answer_words if w in ['autonomous', 'network', 'automation', 'management', 'level', 'intent', 'policy', 'orchestration']]
        
        if not important_words:
            important_words = [w.title() for w in answer_words if len(w) > 4][:6]
        
        if important_words:
            unique_words = list(dict.fromkeys(important_words))[:6]  # Remove duplicates, keep order
            G.add_nodes_from([w.title() for w in unique_words])
            
            # Create connections based on co-occurrence
            nodes = list(G.nodes())
            if len(nodes) > 1:
                center = nodes[0]
                for node in nodes[1:]:
                    G.add_edge(center, node)
        else:
            # Absolute fallback
            G.add_nodes_from(["Question", "Answer", "Analysis"])
            G.add_edges_from([("Question", "Answer"), ("Answer", "Analysis")])
        
        title = "Concept Overview"
        layout_func = nx.circular_layout
        node_color = "lightyellow"
    
    # Generate the plot with enhanced styling
    try:
        fig, ax = plt.subplots(figsize=(14, 10))  # Larger figure
        
        # Apply layout
        try:
            pos = layout_func(G, seed=42)
        except:
            pos = nx.spring_layout(G, seed=42, k=3, iterations=50)
        
        # Calculate node sizes based on importance
        node_count = G.number_of_nodes()
        if node_count > 0:
            base_size = max(2000, 8000 // node_count)
            sizes = []
            for node in G.nodes():
                degree = G.degree(node)
                # Larger nodes for higher degree or important keywords
                importance_boost = 1.5 if any(kw in str(node).lower() for kw in ['level', 'autonomous', 'network']) else 1.0
                size = int(base_size * (1 + degree * 0.3) * importance_boost)
                sizes.append(min(size, 4000))  # Cap maximum size
            
            # Draw nodes with enhanced styling
            nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=sizes, 
                                 ax=ax, alpha=0.8, edgecolors='black', linewidths=2)
            
            # Draw edges with better styling
            if G.number_of_edges() > 0:
                edge_width = 3 if node_count < 6 else 2
                nx.draw_networkx_edges(G, pos, edge_color="darkgray", width=edge_width, 
                                     alpha=0.7, arrows=isinstance(G, nx.DiGraph), 
                                     arrowsize=25, ax=ax, arrowstyle='->')
            
            # Draw labels with better formatting
            font_size = max(8, min(12, 100 // node_count))
            nx.draw_networkx_labels(G, pos, font_size=font_size, font_weight="bold", 
                                  font_color="black", ax=ax)
        
        # Enhanced title
        question_preview = question[:60] + "..." if len(question) > 60 else question
        ax.set_title(f"{title}\n\nQ: {question_preview}", 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add metadata text
        ax.text(0.02, 0.02, f"Generated from: {len(answer.split())} word answer", 
               transform=ax.transAxes, fontsize=10, alpha=0.7,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.axis("off")
        
        # Save to buffer
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight", 
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        plt.close(fig)
        
        return buf
        
    except Exception as e:
        st.error(f"Diagram generation failed: {e}")
        # Create a simple fallback diagram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Diagram Generation\nQuestion: {question[:50]}...\nAnswer: {len(answer)} characters", 
               ha='center', va='center', fontsize=12, 
               bbox=dict(boxstyle="round,pad=1", facecolor="lightgray"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return buf

def generate_audio_transcript():
    """Fixed audio generation with better error handling"""
    if not st.session_state.chat:
        return None
    text = "Autonomous Network Copilot Session Transcript. "
    for i, e in enumerate(st.session_state.chat, 1):
        if e['from'] == 'User' or e['to'] == 'User':  # Only include user-relevant messages
            text += f"Question {i}: {e['question']}. Answer: {e['answer'][:200]}. "
    
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.setProperty("volume", 0.8)
        out = os.path.join(tempfile.gettempdir(), f"transcript_{int(time.time())}.wav")
        engine.save_to_file(text[:1000])  # Limit length
        engine.runAndWait()
        return out
    except Exception as ex:
        st.error(f"Audio generation failed: {ex}")
        return None

# ==================================================
#                  INDEX (QDRANT FIRST) - FIXED
# ==================================================
def build_index(docs_dir: str, use_qdrant=True, collection="multiagent_rag"):
    """COMPLETELY LOCAL index building - NO OPENAI EVER"""
    if SimpleDirectoryReader is None or VectorStoreIndex is None:
        raise RuntimeError("Install llama-index: pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama")
    
    # Load documents quietly
    docs = SimpleDirectoryReader(docs_dir).load_data()
    if not docs:
        raise RuntimeError("No documents found or failed to load documents.")
    
    # Show clean info without temp paths
    st.success(f"📄 Document loaded: {len(docs)} chunks, {sum(len(doc.text) for doc in docs):,} characters")
    
    # FORCE LOCAL EMBEDDING - NO EXCEPTIONS
    try:
        embed_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434",
            request_timeout=120
        )
        
        # TEST embedding first
        test_embed = embed_model.get_text_embedding("test embedding")
        if len(test_embed) < 10:
            raise Exception("Embedding test failed")
            
        st.success("✅ Local embedding model ready")
        
    except Exception as e:
        st.error(f"❌ Embedding model failed: {e}")
        with st.expander("🔧 Fix Embedding Issues"):
            st.code("""
# Make sure Ollama is running and model exists:
ollama serve
ollama pull nomic-embed-text
ollama list  # verify nomic-embed-text is there
            """)
        raise RuntimeError(f"Local embedding failed: {e}")

    # Try Qdrant first
    if use_qdrant and qdrant_client and QdrantVectorStore and StorageContext:
        try:
            client = qdrant_client.QdrantClient(url="http://localhost:6333", prefer_grpc=False, timeout=10)
            client.get_collections()
            vstore = QdrantVectorStore(client=client, collection_name=collection)
            storage = StorageContext.from_defaults(vector_store=vstore)
            
            # EXPLICIT LOCAL SETTINGS
            index = VectorStoreIndex.from_documents(
                docs, 
                storage_context=storage, 
                embed_model=embed_model,
                show_progress=False  # Quiet
            )
            st.success("✅ Qdrant vector store ready")
            return index, True
            
        except Exception as e:
            st.warning(f"Qdrant unavailable: {str(e)[:50]}... Using memory store")
    
    # Fallback to memory with EXPLICIT local settings
    try:
        # FORCE no global defaults
        index = VectorStoreIndex.from_documents(
            docs, 
            embed_model=embed_model,
            show_progress=False,
            # Explicit local settings
            service_context=None  # Don't use global
        )
        st.success("✅ Memory vector store ready")
        return index, False
        
    except Exception as e:
        st.error(f"❌ Even memory indexing failed: {e}")
        raise RuntimeError(f"All indexing methods failed: {e}")

# ==================================================
#                  AGENT RUNTIME - FIXED
# ==================================================
def allow_edge(src, dst): 
    return (src, dst) in st.session_state.edges

def init_llm(model_name: str, timeout_s=300, num_predict=1024):
    """Enhanced LLM initialization with local Ollama - NO API KEYS NEEDED"""
    try:
        # Force local Ollama configuration
        llm = Ollama(
            model=model_name, 
            base_url="http://localhost:11434",  # Explicit Ollama URL
            request_timeout=timeout_s, 
            temperature=0.2,
            num_predict=num_predict, 
            top_k=40, 
            top_p=0.9,
            repeat_penalty=1.1
        )
        
        embed = OllamaEmbedding(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434",  # Explicit Ollama URL
            request_timeout=120
        )
        
        # Test the connection first
        try:
            test_response = llm.complete("Hello")
            st.success(f"✅ Ollama {model_name} connected successfully!")
        except Exception as test_error:
            st.error(f"❌ Ollama connection test failed: {test_error}")
            st.info("Make sure Ollama is running: `ollama serve` and model is available: `ollama pull llama3`")
            return None
        
        # Set global settings ONLY if connection works
        if hasattr(Settings, 'llm'):
            Settings.llm = llm
            Settings.embed_model = embed
            # Disable OpenAI fallback completely
            Settings.chunk_size = 1024
            Settings.chunk_overlap = 20
        
        return llm
    except Exception as e:
        st.error(f"Failed to initialize local Ollama LLM: {e}")
        st.info("💡 **Setup Instructions:**")
        st.code("""
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Start Ollama service
ollama serve

# 3. Pull your model (in another terminal)
ollama pull llama3

# 4. Verify it's working
ollama list
        """)
        return None

# ==================================================
#           FIXED AGENT RETRIEVER - ENHANCED
# ==================================================
def agent_retriever(question: str) -> Tuple[str, str]:
    """Enhanced retrieval with better query processing and context extraction"""
    idx = st.session_state.index
    if idx is None:
        return "[Retriever] No index loaded.", ""
    
    try:
        # Create query engine with enhanced settings for better retrieval
        qeng = idx.as_query_engine(
            similarity_top_k=st.session_state.k * 2,  # Retrieve more chunks initially
            response_mode="no_text",  # Only get source nodes, no LLM response
            verbose=False
        )
        
        # Enhanced query - add context keywords for better matching
        enhanced_query = f"{question} autonomous network levels automation management"
        
        # Query for relevant chunks
        resp = qeng.query(enhanced_query)
        
        # Extract and process source nodes
        source_nodes = getattr(resp, "source_nodes", [])
        
        if source_nodes:
            # Sort by relevance score and take top k
            sorted_nodes = sorted(source_nodes, key=lambda x: getattr(x, 'score', 0), reverse=True)
            top_nodes = sorted_nodes[:st.session_state.k]
            
            # Create comprehensive context with better formatting
            context_parts = []
            for i, node in enumerate(top_nodes):
                chunk_text = node.text.strip()
                if chunk_text:  # Only include non-empty chunks
                    context_parts.append(f"[Document Section {i+1}]\n{chunk_text}")
            
            raw_context = "\n\n".join(context_parts)
            summary = f"[Retriever] Retrieved {len(context_parts)} relevant document sections with {len(raw_context)} characters"
            
            if st.session_state.debug_mode:
                st.markdown(f'<div class="status-box debug-box">📄 Retrieved {len(context_parts)} sections, {len(raw_context)} chars total</div>', 
                           unsafe_allow_html=True)
        else:
            raw_context = ""
            summary = "[Retriever] No relevant context found in documents"
        
        return summary, raw_context
        
    except Exception as e:
        error_msg = f"[Retriever] Error: {str(e)}"
        st.error(f"Retrieval failed: {e}")
        return error_msg, ""

# ==================================================
#           FIXED AGENT RESPONDER - ENHANCED
# ==================================================
def agent_responder(question: str, context: str) -> Tuple[str, str]:
    """Enhanced responder with better prompt handling and response generation"""
    if not context.strip():
        return 'Based on the provided documents, I don\'t have sufficient information to answer this question fully.', ""
    
    # Initialize LLM with enhanced settings
    try:
        llm = Ollama(
            model=st.session_state.model, 
            base_url="http://localhost:11434",
            request_timeout=300, 
            temperature=0.1,  # Lower temperature for more focused responses
            num_predict=2048,  # Increased for longer responses
            top_k=40, 
            top_p=0.85,  # Adjusted for better coherence
            repeat_penalty=1.15,  # Higher to reduce repetition
            system="You are a technical expert in autonomous networks. Provide detailed, accurate responses based on the given context."
        )
    except Exception as e:
        return f"[Responder] LLM initialization failed: {str(e)}", ""
    
    # Enhanced context processing - truncate intelligently
    if len(context) > 6000:
        # Try to keep complete sections
        sections = context.split('[Document Section')
        truncated_sections = []
        current_length = 0
        
        for section in sections:
            section_text = '[Document Section' + section if section != sections[0] else section
            if current_length + len(section_text) < 6000:
                truncated_sections.append(section_text)
                current_length += len(section_text)
            else:
                break
        
        context = "\n".join(truncated_sections)
        if st.session_state.debug_mode:
            st.markdown('<div class="status-box debug-box">✂️ Context truncated to fit model limits</div>', unsafe_allow_html=True)
    
    # Format the enhanced prompt
    formatted_prompt = PROMPT_STRICT.format(
        context_str=context,
        query_str=question
    )
    
    try:
        # Generate response with retries
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = llm.complete(formatted_prompt)
                answer = response.text if hasattr(response, 'text') else str(response)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)  # Brief pause before retry
        
        # Enhanced answer processing
        answer = answer.strip()
        
        # Remove common unwanted prefixes/suffixes
        unwanted_starts = [
            "Based on the provided context,",
            "According to the context,", 
            "From the given information,",
            "The context shows that",
            "Looking at the provided documents,"
        ]
        
        for start in unwanted_starts:
            if answer.lower().startswith(start.lower()):
                answer = answer[len(start):].strip()
        
        # Ensure we have a substantial answer
        if len(answer) < 50 or not answer:
            return 'Based on the provided documents, I don\'t have sufficient information to answer this question fully.', context
        
        # Check for repetitive content
        sentences = answer.split('. ')
        if len(sentences) > 3:
            unique_sentences = []
            seen = set()
            for sentence in sentences:
                sentence_clean = sentence.lower().strip()
                if sentence_clean not in seen and len(sentence_clean) > 10:
                    unique_sentences.append(sentence)
                    seen.add(sentence_clean)
            if len(unique_sentences) < len(sentences) * 0.7:  # If more than 30% repetition
                answer = '. '.join(unique_sentences)
        
        if st.session_state.debug_mode:
            st.markdown(f'<div class="status-box debug-box">💬 Generated answer: {len(answer)} characters, {len(answer.split())} words</div>', 
                       unsafe_allow_html=True)
        
        return answer, context
        
    except Exception as e:
        error_msg = f"[Responder] Error generating response: {str(e)}"
        st.error(error_msg)
        return error_msg, context

# ==================================================
#           FIXED AGENT SUPERVISOR - SIMPLIFIED
# ==================================================
def agent_supervisor_route(user_question: str) -> str:
    """Simplified and fixed agent routing"""
    
    if st.session_state.debug_mode:
        st.markdown('<div class="status-box debug-box">🎯 Starting agent workflow...</div>', unsafe_allow_html=True)
    
    # Step 1: Retrieval
    if st.session_state.debug_mode:
        st.markdown('<div class="status-box debug-box">📖 Step 1: Document retrieval...</div>', unsafe_allow_html=True)
    
    retriever_summary, raw_context = agent_retriever(user_question)
    
    if not raw_context.strip():
        return "Based on the provided documents, I don't have sufficient information to answer this question fully."
    
    # Store context for diagram generation
    st.session_state.last_context = raw_context
    
    # Step 2: Response generation
    if st.session_state.debug_mode:
        st.markdown('<div class="status-box debug-box">🤖 Step 2: Answer generation...</div>', unsafe_allow_html=True)
    
    final_answer, used_context = agent_responder(user_question, raw_context)
    
    # Single comprehensive log entry
    _log("User", "System", user_question, final_answer, used_context)
    
    if st.session_state.debug_mode:
        st.markdown('<div class="status-box debug-box">✅ Workflow completed successfully</div>', unsafe_allow_html=True)
    
    return final_answer

def _log(frm, to, q, a, ctx):
    """Enhanced logging with deduplication"""
    # Avoid duplicate logging
    if st.session_state.chat:
        last_entry = st.session_state.chat[-1]
        if (last_entry['from'] == frm and last_entry['to'] == to and 
            last_entry['question'] == q and last_entry['answer'] == a):
            return  # Skip duplicate
    
    st.session_state.chat.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "from": frm, 
        "to": to, 
        "question": q, 
        "answer": a, 
        "context": ctx[:500] + "..." if len(ctx) > 500 else ctx  # Truncate long context
    })

# ==================================================
#                      SIDEBAR - ENHANCED
# ==================================================
with st.sidebar:
    st.header("🔧 Configuration")
    st.session_state.model = st.selectbox("Ollama model", ["llama3","llama3.1","llama3.2","mistral","qwen2","phi3"], index=0)
    st.session_state.k = st.slider("Retriever: top-k chunks", 1, 10, 5)
    st.session_state.debug_mode = st.checkbox("Enable debug mode", value=True)
    use_qdrant = st.checkbox("Use Qdrant (if available)", value=True)

    st.markdown("---")
    st.header("🧠 Agents")
    st.caption("Current topology:")
    for edge in st.session_state.edges:
        st.write(f"• {edge[0]} → {edge[1]}")
    
    # Quick topology reset
    if st.button("🔄 Reset to Default Topology"):
        st.session_state.edges = {("Supervisor","Retriever"), ("Retriever","Responder"), ("Responder","Supervisor")}
        st.rerun()

    st.markdown("---")
    st.header("📊 Session")
    st.metric("Messages", len(st.session_state.chat))
    st.metric("Index Status", "✅ Ready" if st.session_state.index else "❌ No Index")
    if st.session_state.used_qdrant:
        st.success("🗄️ Using Qdrant")
    elif st.session_state.index:
        st.info("🧠 Using Memory")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat = []
        st.rerun()

# ==================================================
#              UPLOAD & INDEX - ENHANCED
# ==================================================
st.header("📁 Document Upload & Indexing")
files = st.file_uploader("Upload autonomous network documents", accept_multiple_files=True, type=["txt","pdf","docx","md"])

if files and not st.session_state.docs_loaded:
    with st.spinner("Processing document..."):
        d = tempfile.mkdtemp()
        file_names = []
        for f in files:
            file_path = os.path.join(d, f.name)
            with open(file_path, "wb") as o: 
                o.write(f.getbuffer())
            file_names.append(f.name)
        
        st.session_state.docs_dir = d
        st.session_state.docs_loaded = True
        
        # Clean success message without temp path
        st.markdown(f'<div class="status-box success-box">✅ Uploaded: {", ".join(file_names)}</div>', unsafe_allow_html=True)

if st.session_state.docs_loaded and st.session_state.index is None:
    st.markdown('<div class="status-box info-box">🔄 Creating vector index with local Ollama...</div>', unsafe_allow_html=True)
    
    with st.spinner("Building vector index..."):
        try:
            # Build index with completely local settings
            idx, used_q = build_index(st.session_state.docs_dir, use_qdrant=use_qdrant)
            st.session_state.index = idx
            st.session_state.used_qdrant = used_q
            
        except Exception as e:
            st.markdown(f'<div class="status-box warn-box">⚠️ Index creation failed</div>', unsafe_allow_html=True)
            with st.expander("🔧 Troubleshooting"):
                st.error(str(e))
                st.markdown("**Quick Fixes:**")
                st.code("""
# 1. Start Ollama
ollama serve

# 2. Install required model
ollama pull nomic-embed-text

# 3. Test Ollama
curl http://localhost:11434/api/tags

# 4. Restart this app
                """)

# ==================================================
#               USER QUERY INTERFACE - ENHANCED
# ==================================================
st.markdown("---")
st.header("💬 Ask Your Question")

# Sample questions for autonomous networks
sample_questions = [
    "Which are the 6 autonomous network levels mentioned in the document?",
    "What is the difference between Level 0 and Level 5 autonomous networks?",
    "How does intent-based networking work in autonomous systems?",
    "What are the key components of network automation?",
    "Explain the role of AI/ML in autonomous networks."
]

selected_sample = st.selectbox("Try a sample question:", [""] + sample_questions, index=0)

q = st.text_input("Or ask your own question:", 
                  value=selected_sample if selected_sample else "",
                  placeholder="e.g., Explain the autonomous network architecture levels")

col1, col2 = st.columns([3, 1])
with col1:
    ask_button = st.button("🎯 Get Answer", type="primary")
with col2:
    if st.button("🔄 Clear"):
        q = ""
        st.rerun()

if ask_button and q:
    if st.session_state.index is None:
        st.warning("⚠️ Please upload and index documents first.")
    else:
        with st.spinner("Processing your question through the multi-agent system..."):
            try:
                final_answer = agent_supervisor_route(q)
                
                st.markdown("### 🤖 Answer")
                st.markdown(f'<div class="chat-container">{final_answer}</div>', unsafe_allow_html=True)
                
                # Auto-generate visual if we got a good answer
                if final_answer and "don't have sufficient information" not in final_answer:
                    st.session_state.show_visual = True
                    
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                if st.session_state.debug_mode:
                    st.exception(e)

# ==================================================
#                     ENHANCED ACTIONS
# ==================================================
st.markdown("---")
st.subheader("📊 Quick Actions")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🖼️ Generate Diagram"):
        st.session_state.show_visual = True

with col2:
    if st.button("📝 Text Transcript"):
        st.session_state.show_text_download = True

with col3:
    if st.button("🎧 Audio Transcript"):
        st.session_state.show_audio_download = True

with col4:
    if st.button("🔍 Show Dev Console"):
        st.session_state.show_dev_console = not st.session_state.show_dev_console

# ==================================================
#      DEVELOPER CONSOLE (ENHANCED DEBUG)
# ==================================================
if st.session_state.show_dev_console:
    st.markdown("---")
    st.header("🔧 Developer Console")
    
    tab1, tab2, tab3 = st.tabs(["Agent Graph", "Manual Testing", "System Status"])
    
    with tab1:
        st.subheader("Agent Communication Graph")
        G = nx.DiGraph()
        [G.add_node(n) for n in st.session_state.agents.keys()]
        [G.add_edge(s,d) for s,d in st.session_state.edges]
        
        buf = BytesIO()
        plt.figure(figsize=(8,6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", 
                font_size=12, font_weight='bold', arrows=True, arrowsize=20)
        plt.title("Multi-Agent Communication Topology")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=200)
        buf.seek(0)
        plt.close()
        st.image(buf)
        st.download_button("📥 Download Agent Graph", buf.getvalue(), "agent_graph.png", "image/png")
    
    with tab2:
        st.subheader("Manual Agent Testing")
        col_test1, col_test2 = st.columns(2)
        
        with col_test1:
            test_agent = st.selectbox("Test Agent", ["Retriever", "Responder"])
            test_query = st.text_area("Test Query", "What are autonomous network levels?")
            
            if st.button("🧪 Test Agent"):
                if test_agent == "Retriever" and st.session_state.index:
                    summary, context = agent_retriever(test_query)
                    st.write("**Summary:**", summary)
                    st.write("**Context Preview:**", context[:500] + "..." if len(context) > 500 else context)
                elif test_agent == "Responder":
                    context = st.session_state.last_context or "No context available"
                    answer, _ = agent_responder(test_query, context)
                    st.write("**Answer:**", answer)
                else:
                    st.warning("Index not available or agent not supported")
        
        with col_test2:
            st.subheader("Context Inspector")
            if st.session_state.last_context:
                st.text_area("Last Retrieved Context", st.session_state.last_context, height=200)
                st.metric("Context Length", f"{len(st.session_state.last_context)} chars")
            else:
                st.info("No context retrieved yet")
    
    with tab3:
        st.subheader("System Status")
        status_data = {
            "Ollama Model": st.session_state.model,
            "Vector Store": "Qdrant" if st.session_state.used_qdrant else "In-Memory",
            "Documents Loaded": st.session_state.docs_loaded,
            "Index Ready": st.session_state.index is not None,
            "Chat Messages": len(st.session_state.chat),
            "Active Edges": len(st.session_state.edges)
        }
        
        for key, value in status_data.items():
            if isinstance(value, bool):
                st.metric(key, "✅ Yes" if value else "❌ No")
            else:
                st.metric(key, str(value))

# ==================================================
#           ENHANCED CONVERSATION HISTORY
# ==================================================
st.markdown("---")
st.header("📜 Conversation History")

if st.session_state.chat:
    # Filter options
    show_all = st.checkbox("Show all agent messages", value=False)
    
    # Display messages
    displayed_messages = st.session_state.chat if show_all else [
        msg for msg in st.session_state.chat if msg['from'] == 'User' or msg['to'] == 'User' or 'System' in [msg['from'], msg['to']]
    ]
    
    for i, e in enumerate(displayed_messages[-20:], 1):  # Show last 20
        with st.expander(f"💬 {e['timestamp']} — {e['from']} → {e['to']}", expanded=False):
            st.write(f"**Question:** {e['question']}")
            st.write(f"**Answer:** {e['answer']}")
            if e.get('context') and st.session_state.debug_mode:
                with st.expander("🔍 Context Used"):
                    st.text(e['context'])
else:
    st.info("No conversation history yet. Ask a question to get started!")

# ==================================================
#        ACTION HANDLERS (ENHANCED)
# ==================================================
if st.session_state.show_visual:
    st.markdown("---")
    st.subheader("🎨 Dynamic Network Diagram")
    
    if st.session_state.chat:
        # Find the most recent user question and answer
        user_messages = [msg for msg in st.session_state.chat if msg['from'] == 'User' or msg['to'] == 'User']
        if user_messages:
            last_msg = user_messages[-1]
            try:
                with st.spinner("Generating intelligent network diagram..."):
                    img_buffer = create_dynamic_network_diagram(
                        last_msg["answer"], 
                        last_msg.get("context", ""), 
                        last_msg["question"]
                    )
                    st.image(img_buffer, use_container_width=True, caption="Auto-generated Network Diagram")
                    st.download_button(
                        "📥 Download Diagram (PNG)", 
                        data=img_buffer.getvalue(), 
                        file_name=f"network_diagram_{int(time.time())}.png", 
                        mime="image/png"
                    )
            except Exception as e:
                st.error(f"Diagram generation failed: {str(e)}")
                if st.session_state.debug_mode:
                    st.exception(e)
        else:
            st.info("No user conversation to visualize yet.")
    else:
        st.info("No conversation history available for diagram generation.")
    
    st.session_state.show_visual = False

if st.session_state.show_text_download:
    st.markdown("---")
    st.subheader("📄 Session Transcript")
    
    # Create comprehensive transcript
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    header = f"""
Multi-Agent Autonomous Network Copilot - Session Transcript
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {st.session_state.model}
Vector Store: {'Qdrant' if st.session_state.used_qdrant else 'In-Memory'}
Total Messages: {len(st.session_state.chat)}

========================================
CONVERSATION LOG
========================================

"""
    
    transcript_text = header
    for i, e in enumerate(st.session_state.chat, 1):
        transcript_text += f"""
[{i}] {e['timestamp']} | {e['from']} → {e['to']}
Question: {e['question']}
Answer: {e['answer']}
Context: {e.get('context', 'N/A')[:300]}{'...' if len(e.get('context', '')) > 300 else ''}

{'-' * 60}
"""
    
    st.text_area("Transcript Preview", transcript_text[:2000] + "\n\n[... truncated for preview ...]", height=300)
    st.download_button(
        "📥 Download Full Transcript", 
        data=transcript_text.encode("utf-8"), 
        file_name=f"copilot_transcript_{timestamp}.txt", 
        mime="text/plain"
    )
    st.session_state.show_text_download = False

if st.session_state.show_audio_download:
    st.markdown("---")
    st.subheader("🎵 Audio Transcript")
    
    with st.spinner("Generating audio transcript..."):
        audio_path = generate_audio_transcript()
        
    if audio_path and os.path.exists(audio_path):
        try:
            with open(audio_path, "rb") as f: 
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/wav")
            st.download_button(
                "📥 Download Audio Transcript", 
                data=audio_bytes, 
                file_name=f"transcript_{int(time.time())}.wav", 
                mime="audio/wav"
            )
            # Clean up temp file
            try: 
                os.unlink(audio_path)
            except: 
                pass
        except Exception as e:
            st.error(f"Audio processing failed: {e}")
    else:
        st.warning("Audio generation failed. Please check your pyttsx3 installation.")
    
    st.session_state.show_audio_download = False


# ==================================================
#                       FOOTER
# ==================================================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**🤖 Multi-Agent System**")
    st.caption("Supervisor → Retriever → Responder")

with footer_col2:
    st.markdown("**🗄️ Vector Storage**")
    st.caption("Qdrant + LlamaIndex RAG")

with footer_col3:
    st.markdown("**🧠 Local AI**")
    st.caption("Ollama + Autonomous Networks")

st.markdown(
    "<div style='text-align:center; color:#9aa0a6; margin-top: 2rem;'>Multi-Agent Autonomous Network Copilot v2.0 (Debug Enhanced)</div>",
    unsafe_allow_html=True
)