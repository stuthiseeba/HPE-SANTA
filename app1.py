import streamlit as st
import os
import tempfile
import pyttsx3
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
import subprocess
import threading
import time
from datetime import datetime
from io import BytesIO
import base64
import re
from typing import List, Dict, Tuple
from collections import Counter

# LlamaIndex imports
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# ===================== Streamlit Setup =====================
st.set_page_config(
    page_title="🤖 Autonomous Network Copilot", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .action-buttons {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    .status-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .answer-box {
        color: black !important;
    }     
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🤖 Autonomous Network Copilot</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6c757d;">Your intelligent assistant for autonomous network documentation</p>', unsafe_allow_html=True)

# ===================== Enhanced Prompts =====================
CONTEXT_PROMPT_TEMPLATE = PromptTemplate(
    """You are an expert assistant for autonomous network documentation. 

    CRITICAL INSTRUCTIONS:
    1. You MUST base your answer ONLY on the provided context from the uploaded documents
    2. If the context doesn't contain enough information, clearly state "Based on the provided documents, I don't have sufficient information to answer this question fully."
    3. Do NOT use external knowledge or make assumptions beyond what's in the context
    4. Keep your response concise and focused (under 300 words)
    5. Be specific and technical when the context allows it

    Context information:
    ---------------------
    {context_str}
    ---------------------

    Question: {query_str}

    Provide a focused answer based strictly on the provided context (max 300 words):"""
)

# ===================== Sidebar Configuration =====================
with st.sidebar:
    st.header("🔧 Configuration")
    model_choice = st.selectbox(
        "Choose LLM Model:", 
        ["llama3", "mistral", "tinyllama"], 
        index=0,
        help="Select the local Ollama model for responses"
    )
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        similarity_k = st.slider("Document Chunks to Retrieve", 1, 10, 3, 
                                help="Number of relevant document sections to use (lower = faster)")
        context_strict = st.checkbox("Strict Context Mode", value=True,
                                   help="Enforce strict adherence to document context")
        response_timeout = st.slider("Response Timeout (seconds)", 60, 600, 300,
                                   help="Maximum time to wait for model response")
        max_response_length = st.slider("Max Response Length", 256, 2048, 512,
                                      help="Limit response length for faster processing")
    
    st.markdown("---")
    st.header("📊 Session Info")
    if "chat_history" in st.session_state and st.session_state.chat_history:
        st.metric("Total Questions", len(st.session_state.chat_history))
        st.metric("Session Duration", f"{int((time.time() - st.session_state.get('session_start', time.time()))/60)} min")

# ===================== Session State Initialization =====================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "continue_chat" not in st.session_state:
    st.session_state.continue_chat = True
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "current_answer" not in st.session_state:
    st.session_state.current_answer = ""
if "current_context" not in st.session_state:
    st.session_state.current_context = ""
if "awaiting_action" not in st.session_state:
    st.session_state.awaiting_action = False
if "session_start" not in st.session_state:
    st.session_state.session_start = time.time()

# ===================== Enhanced Helper Functions =====================
def extract_network_entities_from_context(answer_text: str, context_text: str) -> Dict[str, List[str]]:
    """Extract network-related entities from the answer and context for diagram generation"""
    
    # Combine answer and context for analysis
    full_text = f"{answer_text} {context_text}".lower()
    
    entities = {
        'nodes': [],
        'connections': [],
        'layers': [],
        'protocols': [],
        'components': []
    }
    
    # Network nodes/devices patterns
    node_patterns = [
        r'\b(?:router|switch|server|controller|gateway|firewall|load balancer|proxy|hub|bridge|node|device|endpoint|host|client|workstation)\b',
        r'\b(?:core|edge|access|distribution|aggregation|spine|leaf)\s+(?:router|switch|node)',
        r'\b(?:sdn|software.defined.network)\s+controller\b',
        r'\b(?:management|control|data)\s+plane\b'
    ]
    
    # Connection/relationship patterns
    connection_patterns = [
        r'(\w+)\s+(?:connects to|connected to|linked to|communicates with|sends to|receives from)\s+(\w+)',
        r'(\w+)\s+(?:→|->|flows to|routes to)\s+(\w+)',
        r'between\s+(\w+)\s+and\s+(\w+)'
    ]
    
    # Layer patterns
    layer_patterns = [
        r'\b(?:physical|data link|network|transport|session|presentation|application)\s+layer\b',
        r'\bl[0-7]\b',
        r'\b(?:layer\s+[0-7]|layer\s+(?:one|two|three|four|five|six|seven))\b'
    ]
    
    # Protocol patterns
    protocol_patterns = [
        r'\b(?:tcp|udp|ip|http|https|ftp|dns|dhcp|snmp|bgp|ospf|eigrp|rip|stp|vlan|mpls|vpn|ssl|tls)\b',
        r'\b(?:ethernet|wifi|802\.11|bluetooth|zigbee)\b'
    ]
    
    # Component patterns
    component_patterns = [
        r'\b(?:database|api|service|application|interface|module|engine|processor|memory|storage)\b',
        r'\b(?:microservice|container|virtualization|orchestration)\b'
    ]
    
    # Extract entities
    for pattern in node_patterns:
        matches = re.findall(pattern, full_text)
        entities['nodes'].extend([match.title().replace('_', ' ') for match in matches])
    
    for pattern in connection_patterns:
        matches = re.findall(pattern, full_text)
        entities['connections'].extend([(m[0].title(), m[1].title()) for m in matches if len(m) == 2])
    
    for pattern in layer_patterns:
        matches = re.findall(pattern, full_text)
        entities['layers'].extend([match.title() for match in matches])
    
    for pattern in protocol_patterns:
        matches = re.findall(pattern, full_text)
        entities['protocols'].extend([match.upper() for match in matches])
    
    for pattern in component_patterns:
        matches = re.findall(pattern, full_text)
        entities['components'].extend([match.title() for match in matches])
    
    # Remove duplicates and clean up
    for key in entities:
        entities[key] = list(set(entities[key]))
        # Remove empty strings and single characters
        entities[key] = [item for item in entities[key] if len(item) > 1]
    
    return entities

def create_dynamic_network_diagram(answer_text: str, context_text: str, question: str) -> BytesIO:
    """Create network diagrams based on the actual answer content and context"""
    
    # Extract entities from the answer and context
    entities = extract_network_entities_from_context(answer_text, context_text)
    
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    G = nx.DiGraph() if any(entities['connections']) else nx.Graph()
    
    # Determine diagram type and build graph
    if entities['layers'] and len(entities['layers']) > 2:
        # Layer-based diagram
        G, title, layout_func = create_layered_diagram(entities)
        node_color = 'lightgreen'
    elif entities['connections']:
        # Connection-based diagram
        G, title = create_connection_diagram(entities)
        layout_func = nx.spring_layout
        node_color = 'lightblue'
    elif entities['nodes'] and len(entities['nodes']) > 1:
        # Node-based diagram
        G, title = create_node_diagram(entities)
        layout_func = nx.spring_layout
        node_color = 'lightcoral'
    elif entities['protocols']:
        # Protocol-focused diagram
        G, title = create_protocol_diagram_dynamic(entities)
        layout_func = nx.circular_layout
        node_color = 'gold'
    else:
        # Fallback: create a simple conceptual diagram
        G, title = create_conceptual_diagram(answer_text, question)
        layout_func = nx.spring_layout
        node_color = 'lightsalmon'
    
    if G.number_of_nodes() == 0:
        # Create a simple diagram if no entities found
        G.add_nodes_from(['Query', 'Answer', 'Context'])
        G.add_edges_from([('Query', 'Answer'), ('Context', 'Answer')])
        title = "Query-Answer Relationship"
        layout_func = nx.spring_layout
        node_color = 'lightgray'
    
    # Layout
    try:
        if layout_func == nx.spring_layout:
            pos = layout_func(G, k=3, iterations=50, seed=42)
        else:
            pos = layout_func(G)
    except:
        pos = nx.spring_layout(G, seed=42)
    
    # Draw network
    node_sizes = []
    for node in G.nodes():
        # Size nodes based on degree centrality
        degree = G.degree(node) if hasattr(G, 'degree') else 1
        node_sizes.append(max(2000, degree * 500))
    
    nx.draw_networkx_nodes(G, pos, node_color=node_color, 
                          node_size=node_sizes, alpha=0.8, ax=ax)
    
    # Draw edges
    if G.number_of_edges() > 0:
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              alpha=0.6, width=2, 
                              arrows=isinstance(G, nx.DiGraph), 
                              arrowsize=20, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, 
                           font_weight='bold', ax=ax)
    
    # Title and formatting
    ax.set_title(f"{title}\nBased on: {question[:50]}{'...' if len(question) > 50 else ''}", 
                fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add metadata
    metadata_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    metadata_text += f"Entities: {sum(len(v) for v in entities.values())} found\n"
    metadata_text += f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}"
    
    ax.text(0.02, 0.02, metadata_text, transform=ax.transAxes, 
            fontsize=8, alpha=0.7, verticalalignment='bottom')
    
    plt.tight_layout()
    
    # Save to BytesIO
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    img_buffer.seek(0)
    plt.close(fig)
    
    return img_buffer

def create_layered_diagram(entities: Dict[str, List[str]]) -> Tuple[nx.DiGraph, str, callable]:
    """Create a layered network diagram"""
    G = nx.DiGraph()
    layers = sorted(entities['layers'])
    
    # Add layer nodes
    for i, layer in enumerate(layers):
        G.add_node(layer, level=i)
    
    # Connect adjacent layers
    for i in range(len(layers) - 1):
        G.add_edge(layers[i], layers[i + 1])
    
    # Add protocols to relevant layers
    for protocol in entities['protocols'][:5]:  # Limit to 5 protocols
        G.add_node(protocol, level=-1)
        if layers:
            G.add_edge(protocol, layers[0])
    
    def hierarchical_layout(G):
        pos = {}
        levels = {}
        for node, data in G.nodes(data=True):
            level = data.get('level', 0)
            if level not in levels:
                levels[level] = []
            levels[level].append(node)
        
        for level, nodes in levels.items():
            for i, node in enumerate(nodes):
                pos[node] = (i - len(nodes)/2, -level)
        
        return pos
    
    return G, "Network Layer Architecture", hierarchical_layout

def create_connection_diagram(entities: Dict[str, List[str]]) -> Tuple[nx.DiGraph, str]:
    """Create a connection-based diagram"""
    G = nx.DiGraph()
    
    # Add connections
    for source, target in entities['connections']:
        G.add_edge(source, target)
    
    # Add standalone nodes if any
    for node in entities['nodes']:
        if node not in G.nodes():
            G.add_node(node)
    
    return G, "Network Connections"

def create_node_diagram(entities: Dict[str, List[str]]) -> Tuple[nx.Graph, str]:
    """Create a node-based diagram"""
    G = nx.Graph()
    nodes = entities['nodes'][:10]  # Limit to 10 nodes for clarity
    
    # Add nodes
    G.add_nodes_from(nodes)
    
    # Create a connected graph
    if len(nodes) > 1:
        # Create a hub-spoke or mesh topology
        if len(nodes) <= 4:
            # Full mesh for small networks
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i+1:]:
                    G.add_edge(node1, node2)
        else:
            # Hub-spoke topology
            hub = nodes[0]
            for node in nodes[1:]:
                G.add_edge(hub, node)
    
    return G, "Network Topology"

def create_protocol_diagram_dynamic(entities: Dict[str, List[str]]) -> Tuple[nx.Graph, str]:
    """Create a protocol-focused diagram"""
    G = nx.Graph()
    protocols = entities['protocols'][:8]  # Limit protocols
    
    # Add protocol nodes
    G.add_nodes_from(protocols)
    
    # Group related protocols
    protocol_groups = {
        'Transport': ['TCP', 'UDP'],
        'Network': ['IP', 'BGP', 'OSPF', 'RIP'],
        'Application': ['HTTP', 'HTTPS', 'FTP', 'DNS', 'DHCP'],
        'Data Link': ['ETHERNET', 'WIFI', '802.11']
    }
    
    # Add group nodes and connections
    for group, group_protocols in protocol_groups.items():
        group_members = [p for p in protocols if p in group_protocols]
        if group_members:
            G.add_node(f"{group} Layer")
            for protocol in group_members:
                G.add_edge(f"{group} Layer", protocol)
    
    return G, "Protocol Architecture"

def create_conceptual_diagram(answer_text: str, question: str) -> Tuple[nx.Graph, str]:
    """Create a simple conceptual diagram when specific entities aren't found"""
    G = nx.Graph()
    
    # Extract key concepts from question and answer
    key_words = []
    for text in [question, answer_text]:
        words = re.findall(r'\b\w{4,}\b', text.lower())
        key_words.extend([w.title() for w in words if w not in ['that', 'this', 'with', 'from', 'have', 'been', 'will', 'would', 'could', 'should']])
    
    # Get most frequent concepts
    concepts = Counter(key_words).most_common(6)
    
    if concepts:
        nodes = [concept[0] for concept in concepts]
        G.add_nodes_from(nodes)
        
        # Connect the main concept to others
        if len(nodes) > 1:
            main_concept = nodes[0]
            for concept in nodes[1:]:
                G.add_edge(main_concept, concept)
    else:
        # Fallback
        G.add_nodes_from(['Question', 'Answer', 'Context'])
        G.add_edges_from([('Question', 'Answer'), ('Context', 'Answer')])
    
    return G, "Conceptual Overview"

def generate_audio_transcript():
    """Generate audio from chat history"""
    if not st.session_state.chat_history:
        return None
    
    # Prepare text
    transcript_text = "Autonomous Network Copilot Session Transcript. "
    for i, entry in enumerate(st.session_state.chat_history, 1):
        transcript_text += f"Question {i}: {entry['question']}. "
        transcript_text += f"Answer {i}: {entry['answer']}. "
    
    # Generate audio
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        
        # Create temporary file
        audio_path = os.path.join(tempfile.gettempdir(), f"transcript_{int(time.time())}.wav")
        engine.save_to_file(transcript_text, audio_path)
        engine.runAndWait()
        
        return audio_path
    except Exception as e:
        st.error(f"Audio generation failed: {str(e)}")
        return None

# ===================== File Upload Section =====================
st.header("📁 Document Upload")

uploaded_files = st.file_uploader(
    "Upload your autonomous network documents",
    type=["txt", "pdf", "docx", "md"],
    accept_multiple_files=True,
    help="Supported formats: TXT, PDF, DOCX, Markdown"
)

docs_path = tempfile.mkdtemp()
if uploaded_files and not st.session_state.documents_loaded:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(uploaded_files):
        progress = (i + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"Processing {file.name}...")
        
        file_path = os.path.join(docs_path, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    
    st.session_state.documents_loaded = True
    status_text.empty()
    progress_bar.empty()
    
    st.markdown(f'<div class="status-box success-box">✅ Successfully uploaded {len(uploaded_files)} document(s)</div>', 
                unsafe_allow_html=True)

# ===================== Model Configuration =====================
if st.session_state.documents_loaded:
    with st.spinner(f"🚀 Initializing {model_choice} model..."):
        try:
            # Configure LlamaIndex settings with extended timeouts
            Settings.llm = Ollama(
                model=model_choice, 
                request_timeout=300.0,  # 5 minutes
                temperature=0.1,  # Lower temperature for more focused responses
                num_predict=512,  # Limit response length
                top_k=20,
                top_p=0.9
            )
            Settings.embed_model = OllamaEmbedding(
                model_name="nomic-embed-text",
                request_timeout=120.0
            )
            
            # Warm up model
            try:
                subprocess.run(["ollama", "run", model_choice, "test"], 
                             capture_output=True, text=True, timeout=30)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass  # Continue if warm-up fails
            
        except Exception as e:
            st.error(f"Model initialization error: {e}")
            st.stop()

    # ===================== Build Index =====================
    if 'index' not in st.session_state:
        with st.spinner("📚 Building document index..."):
            try:
                documents = SimpleDirectoryReader(docs_path).load_data()
                st.session_state.index = VectorStoreIndex.from_documents(documents)
                st.markdown('<div class="status-box info-box">📚 Document index ready. You can now ask questions!</div>', 
                           unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Index building failed: {e}")
                st.stop()

    # ===================== Enhanced Question Answering Interface =====================
    st.header("💬 Ask Your Question")
    
    # Display conversation history
    if st.session_state.chat_history:
        st.subheader("📜 Conversation History")
        for i, entry in enumerate(st.session_state.chat_history, 1):
            with st.expander(f"Q{i}: {entry['question'][:50]}..."):
                st.write(f"**Question:** {entry['question']}")
                st.write(f"**Answer:** {entry['answer']}")
                if entry.get('context_used'):
                    with st.expander("View Context Used"):
                        st.text(entry['context_used'][:500] + "..." if len(entry['context_used']) > 500 else entry['context_used'])

    # Main query interface
    if st.session_state.continue_chat:
        query = st.text_input(
            "Enter your question about autonomous networks:",
            placeholder="e.g., What is an autonomous network architecture?",
            key="query_input"
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("🎯 Get Answer", type="primary")

        # Handle fast retry
        if st.session_state.get('retry_with_fast_settings'):
            st.session_state.retry_with_fast_settings = False
            st.info("🚀 Retrying with optimized settings...")
            
            # Force fast settings
            fast_similarity_k = 2
            fast_timeout = 120
            fast_length = 256
            
            with st.spinner("🤔 Quick analysis with reduced settings..."):
                try:
                    # Update to fastest settings
                    Settings.llm = Ollama(
                        model="tinyllama",  # Force fastest model
                        request_timeout=fast_timeout,
                        temperature=0.1,
                        num_predict=fast_length,
                        top_k=10,
                        top_p=0.8
                    )
                    
                    query_engine = st.session_state.index.as_query_engine(
                        similarity_top_k=fast_similarity_k,
                        text_qa_template=CONTEXT_PROMPT_TEMPLATE
                    )
                    
                    response = query_engine.query(query)
                    st.session_state.current_answer = response.response
                    st.session_state.current_context = " ".join([node.text for node in response.source_nodes]) if hasattr(response, 'source_nodes') else ""
                    
                    st.session_state.chat_history.append({
                        "question": query,
                        "answer": st.session_state.current_answer,
                        "context_used": st.session_state.current_context,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    st.session_state.awaiting_action = True
                    st.success("✅ Response generated with fast settings!")
                    
                except Exception as e:
                    st.error(f"Even fast settings failed: {e}")
                    st.markdown("**Troubleshooting:**")
                    st.markdown("1. Check if Ollama is running: `ollama serve`")
                    st.markdown("2. Pull a model: `ollama pull tinyllama`")
                    st.markdown("3. Restart the app")

        if ask_button and query:
            st.session_state.current_question = query
            
            with st.spinner("🤔 Analyzing your question and searching documents..."):
                try:
                    # Update LLM settings with current parameters
                    Settings.llm = Ollama(
                        model=model_choice, 
                        request_timeout=response_timeout,
                        temperature=0.1,
                        num_predict=max_response_length,
                        top_k=20,
                        top_p=0.9
                    )
                    
                    # Create query engine with custom prompt and timeout handling
                    query_engine = st.session_state.index.as_query_engine(
                        similarity_top_k=similarity_k,
                        text_qa_template=CONTEXT_PROMPT_TEMPLATE,
                        streaming=False  # Disable streaming for better timeout handling
                    )
                    
                    # Add progress indicator
                    progress_placeholder = st.empty()
                    progress_placeholder.info("🔍 Searching through documents...")
                    
                    # Execute query
                    progress_placeholder.info("🤖 Generating response...")
                    response = query_engine.query(query)
                    progress_placeholder.empty()
                    
                    st.session_state.current_answer = response.response
                    
                    # Extract context for diagram generation
                    context_chunks = []
                    if hasattr(response, 'source_nodes'):
                        for node in response.source_nodes:
                            context_chunks.append(node.text)
                    
                    st.session_state.current_context = " ".join(context_chunks)
                    
                    # Add to history with context
                    st.session_state.chat_history.append({
                        "question": query,
                        "answer": st.session_state.current_answer,
                        "context_used": st.session_state.current_context,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    st.session_state.awaiting_action = True
                    
                    # Check if answer seems to be based on context
                    if context_strict and ("I don't have sufficient information" in st.session_state.current_answer or 
                                         len(st.session_state.current_context) < 50):
                        st.warning("⚠️ Limited context found. Answer may not be fully based on uploaded documents.")
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    if any(keyword in error_msg for keyword in ["timeout", "timed out", "connection", "network"]):
                        st.error("⏱️ **Timeout Error!** The model took too long to respond.")
                        st.markdown("""
                        **Quick Fixes:**
                        - 🔧 **Reduce settings**: Lower 'Document Chunks' and 'Max Response Length' in sidebar
                        - 🚀 **Switch model**: Try 'tinyllama' for faster responses
                        - ❓ **Shorter question**: Ask more specific, focused questions  
                        - 🔄 **Restart Ollama**: Run `ollama serve` in terminal if needed
                        - 📊 **Check resources**: Ensure your system has enough RAM/CPU
                        """)
                        
                        # Offer to retry with reduced settings
                        if st.button("🔄 Retry with Faster Settings"):
                            st.session_state.retry_with_fast_settings = True
                            st.rerun()
                            
                    else:
                        st.error(f"Error generating response: {e}")
                        st.info("💡 Try rephrasing your question or check if Ollama is running properly.")

        # Display current answer and action options
        if st.session_state.awaiting_action and st.session_state.current_answer:
            st.markdown("---")
            st.subheader("💡 Answer")
            st.markdown(f'<div class="chat-container answer-box">{st.session_state.current_answer}</div>', 
                       unsafe_allow_html=True)

            # Show context information
            if st.session_state.current_context:
                with st.expander("🔍 View Source Context"):
                    st.text_area("Context from documents:", 
                               value=st.session_state.current_context, 
                               height=200, 
                               disabled=True)
            
            st.subheader("🎯 What would you like to do next?")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                if st.button("❓ Ask More Questions", help="Continue the conversation"):
                    st.session_state.awaiting_action = False
                    st.session_state.current_question = ""
                    st.session_state.current_answer = ""
                    st.session_state.current_context = ""
                    st.rerun()
            
            with col2:
                if st.button("📊 Generate Visual", help="Create a dynamic network diagram based on the answer"):
                    st.session_state.show_visual = True
            
            with col3:
                if st.button("📝 Text Transcript", help="Download conversation as text"):
                    st.session_state.show_text_download = True
            
            with col4:
                if st.button("🎵 Audio Transcript", help="Download conversation as audio"):
                    st.session_state.show_audio_download = True
            
            with col5:
                if st.button("✅ End Session", help="Finish and close"):
                    st.session_state.continue_chat = False
                    st.session_state.awaiting_action = False
                    st.rerun()

            # Handle action responses
            if st.session_state.get('show_visual'):
                st.markdown("---")
                st.subheader("📊 Dynamic Network Diagram")
                st.info("🎯 This diagram is generated based on the content of the answer and context from your documents.")
                
                try:
                    with st.spinner("Creating dynamic diagram from answer content..."):
                        diagram_buffer = create_dynamic_network_diagram(
                            st.session_state.current_answer,
                            st.session_state.current_context,
                            st.session_state.current_question
                        )
                    
                    st.image(diagram_buffer, caption="Dynamic Network Diagram (Generated from Answer)", 
                            use_container_width=True)
                    
                    st.download_button(
                        label="📥 Download Diagram",
                        data=diagram_buffer.getvalue(),
                        file_name=f"dynamic_network_diagram_{int(time.time())}.png",
                        mime="image/png"
                    )
                    
                except Exception as e:
                    st.error(f"Diagram generation failed: {e}")
                    st.error("This might happen if the answer doesn't contain enough network-related information for visualization.")
                
                st.session_state.show_visual = False

            if st.session_state.get('show_text_download'):
                st.markdown("---")
                st.subheader("📝 Text Transcript")
                
                transcript_content = f"Autonomous Network Copilot - Session Transcript\n"
                transcript_content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                transcript_content += f"Model Used: {model_choice}\n"
                transcript_content += f"Context Mode: {'Strict' if context_strict else 'Relaxed'}\n"
                transcript_content += "="*50 + "\n\n"
                
                for i, entry in enumerate(st.session_state.chat_history, 1):
                    transcript_content += f"Question {i}: {entry['question']}\n\n"
                    transcript_content += f"Answer {i}: {entry['answer']}\n\n"
                    if entry.get('context_used'):
                        transcript_content += f"Context Used: {entry['context_used'][:200]}...\n\n"
                    transcript_content += "-"*30 + "\n\n"
                
                st.download_button(
                    label="📥 Download Text Transcript",
                    data=transcript_content.encode('utf-8'),
                    file_name=f"copilot_transcript_{int(time.time())}.txt",
                    mime="text/plain"
                )
                
                st.session_state.show_text_download = False

            if st.session_state.get('show_audio_download'):
                st.markdown("---")
                st.subheader("🎵 Audio Transcript")
                
                with st.spinner("Generating audio transcript..."):
                    audio_path = generate_audio_transcript()
                    
                    if audio_path and os.path.exists(audio_path):
                        with open(audio_path, "rb") as audio_file:
                            audio_bytes = audio_file.read()
                        
                        st.audio(audio_bytes, format="audio/wav")
                        
                        st.download_button(
                            label="📥 Download Audio Transcript",
                            data=audio_bytes,
                            file_name=f"copilot_audio_{int(time.time())}.wav",
                            mime="audio/wav"
                        )
                        
                        # Clean up
                        try:
                            os.unlink(audio_path)
                        except:
                            pass
                    else:
                        st.error("Failed to generate audio transcript")
                
                st.session_state.show_audio_download = False

    else:
        st.markdown("---")
        st.markdown('<div class="status-box success-box">✅ Session ended. Thank you for using Autonomous Network Copilot!</div>', 
                   unsafe_allow_html=True)
        
        if st.button("🔄 Start New Session"):
            # Reset session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

else:
    st.markdown('<div class="status-box info-box">📋 Please upload your autonomous network documents to get started.</div>', 
               unsafe_allow_html=True)

# ===================== Footer =====================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #6c757d; padding: 1rem;">
        🤖 Autonomous Network Copilot v2.0 | Enhanced Context Awareness | Local Processing Only
        <br><small>Now with Dynamic Diagram Generation & Strict Context Mode</small>
    </div>
    """,
    unsafe_allow_html=True
)