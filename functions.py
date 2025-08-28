#--Imports--
import streamlit as st
import os
import re
import uuid
import logging
import tempfile
import time
#--logging configuration-
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#--import--
from utils import (
        InMemoryVectorStore, load_document, chunk_document, generate_embeddings,
        save_embeddings, load_embeddings, get_available_embedding_sets,
        generate_structured_answer_with_gemini, embedding_model,
        calculate_retrieval_metrics, calculate_response_metrics,
        calculate_system_performance_metrics, format_metrics_for_display,
        calculate_overall_system_accuracy, iterative_summarize_chunks,
        load_summary_cache, save_summary_cache, load_answer_cache,
        save_answer_cache, keyword_search, hybrid_search, rerank_chunks,
        rewrite_query, compress_context, estimate_tokens, refine_answer_with_gemini
    )
#--imports from voice utils--
try:
    from voice_utils import (
        initialize_voice_state,create_gtts_audio_player,
        create_voice_settings_interface,
        get_system_audio_info,switch_to_js_tts,
        TTS_AVAILABLE, AUDIO_PLAYBACK_AVAILABLE, synthesize_speech,
        record_audio_from_mic, transcribe_audio,create_js_tts_player
    )
    logger.info("Successfully imported voice utilities with JavaScript TTS")
    VOICE_FEATURES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Voice utilities not available: {e}")
    TTS_AVAILABLE = False
    AUDIO_PLAYBACK_AVAILABLE = False
    VOICE_FEATURES_AVAILABLE = False


#--state initializations--
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = InMemoryVectorStore()
if 'embedding_model_loaded' not in st.session_state:
    st.session_state.embedding_model_loaded = embedding_model is not None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_embeddings_name' not in st.session_state:
    st.session_state.current_embeddings_name = None
if 'last_query_metrics' not in st.session_state:
    st.session_state.last_query_metrics = None
if 'last_processed_query' not in st.session_state:
    st.session_state.last_processed_query = ""
if 'query_counter' not in st.session_state:
    st.session_state.query_counter = 0
if VOICE_FEATURES_AVAILABLE:
    initialize_voice_state()

#--initial page configuration--
def config():
    st.set_page_config(
    page_title="RAG Prototype - Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded")

#-css used for styling--
def css():
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
            text-align: center;
        }
        .feature-box {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin: 0.5rem 0;
        }
        .feature-box:hover{
        color: linear-gradient(90deg, #667eea 0%, #764ba2 100%);        
        }
        .status-success {
            color: #28a745;
            font-weight: bold;
        }
        .status-warning {
            color: #ffc107;
            font-weight: bold;
        }
        .status-error {
            color: #dc3545;
            font-weight: bold;
        }
        .audio-section {
            background: #e8f4f8;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .questions-asked-box {
            text-align: center; 
            background: rgba(0,0,0,0);
            color: white;
        }
        .voice-input-container {
            display: flex;
            align-items: flex-end;
            height: 100%;
            padding-top: 1.75rem;
        }
        .voice-input-container button {
            height: 2.5rem !important;
            width: 100% !important;
            margin-bottom: 0.2rem !important;
        }
        div[data-testid="column"] button[key="load_btn"],
        div[data-testid="column"] button[key="delete_btn"] {
            height: 2.5rem !important;
            font-size: 0.9rem !important;
        }
        div[data-testid="column"]:has(button:contains("Voice Input")) {
            display: flex;
            align-items: flex-end;
            padding-top: 1.5rem;
        }
    </style>
    """, unsafe_allow_html=True)


def main_title():
    st.markdown("""
    <div class="main-header">
        <h1>📚Intelligent Document Q&A System</h1>
    </div>
    """, unsafe_allow_html=True)

    # Feature highlights
    with st.expander("🚀 Key Features", expanded=False):
        st.markdown("""
        <div class="feature-box">
            <ul>
                <li>📄 <strong>Multi-format Support:</strong> Upload and process PDF/TXT documents</li>
                <li>🧠 <strong>AI-Powered Search:</strong> Semantic embeddings for intelligent document retrieval</li>
                <li>💬 <strong>Natural Conversation:</strong> Chat with your documents using natural language</li>
                <li>🎤 <strong>Voice Input:</strong> Hands-free queries with speech recognition</li>
                <li>🔊 <strong>Audio Responses:</strong> Browser-based text-to-speech</li>
                <li>📊 <strong>Source Citations:</strong> View exact document references for every answer</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

#--sidebar menu--
def sidebar():
    with st.sidebar:
        st.markdown("## 📄 Document Management")
        st.markdown("### 🔧 System Status")
        system_status = []
        if embedding_model is not None:
            system_status.append("🧠 AI Models: <span class='status-success'>Loaded</span>")
        else:
            system_status.append("🧠 AI Models: <span class='status-error'>Failed</span>")
        
        for status in system_status:
            st.markdown(status, unsafe_allow_html=True)
        
        st.divider()
        
        #--Document Upload Section--
        st.markdown("### 📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF or TXT files",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload multiple documents to create a searchable knowledge base")
        
        new_embedding_set_name = st.text_input(
            "Knowledge Base Name",
            placeholder="e.g., 'Technical Manuals', 'Research Papers'...",
            help="Give your document collection a descriptive name")
        
        if st.button("🚀 Process Documents", disabled=not st.session_state.embedding_model_loaded):
            if uploaded_files and new_embedding_set_name.strip():
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Processing documents..."):
                    all_chunks = []
                    total_files = len(uploaded_files)
                    
                    try:
                        for i, uploaded_file in enumerate(uploaded_files):
                            status_text.text(f"Processing {uploaded_file.name}...")
                            progress_bar.progress((i + 0.5) / total_files)
                            
                            #--Save temporary file--
                            os.makedirs("data", exist_ok=True)
                            temp_file_path = os.path.join("data", uploaded_file.name)
                            
                            with open(temp_file_path, "wb") as f:
                                f.write(uploaded_file.getvalue())
                            
                            #--Process document--
                            doc_id = str(uuid.uuid4())
                            text, structured_pages, toc_data, error = load_document(temp_file_path)
                            
                            if error:
                                st.error(f"Error loading {uploaded_file.name}: {error}")
                                continue
                            
                            chunks = chunk_document(text, doc_id, uploaded_file.name, 
                                                structured_pages=structured_pages, toc_data=toc_data)
                            chunks_with_embeddings = generate_embeddings(chunks)
                            all_chunks.extend(chunks_with_embeddings)            
                            os.remove(temp_file_path)                
                            progress_bar.progress((i + 1) / total_files)
                        
                        #--Save embeddings--
                        if all_chunks:
                            status_text.text("Saving knowledge base...")
                            success, error = save_embeddings(all_chunks, new_embedding_set_name)
                            if success:
                                st.session_state.vector_store = InMemoryVectorStore()
                                st.session_state.vector_store.add_vectors(all_chunks)
                                st.session_state.current_embeddings_name = new_embedding_set_name
                                st.session_state.chat_history = []
                                
                                st.success(f"✅ Successfully processed {len(uploaded_files)} files!")
                                st.info(f"📊 Generated {len(all_chunks)} searchable chunks")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error(f"Error saving knowledge base: {error}")
                        
                        progress_bar.empty()
                        status_text.empty()
                    
                    except Exception as e:
                        st.error(f"Processing error: {e}")
                        progress_bar.empty()
                        status_text.empty()
            else:
                st.error("Please upload files and enter a knowledge base name.")
    
        st.divider()
        
        #--Load Existing Embeddings--
        st.markdown("### 📂 Load Existing Knowledge Base")
        available_sets = get_available_embedding_sets()
        
        if available_sets:
            selected_set = st.selectbox(
                "Choose knowledge base:", 
                available_sets,
                help="Select a previously processed document collection"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Load Selected", use_container_width=True, key="load_btn"):
                    with st.spinner("Loading knowledge base..."):
                        embeddings_data, error = load_embeddings(selected_set)
                        
                        if error:
                            st.error(f"Loading error: {error}")
                        elif not embeddings_data:
                            st.error(f"No valid data in '{selected_set}'")
                        else:
                            st.session_state.vector_store = InMemoryVectorStore()
                            st.session_state.vector_store.add_vectors(embeddings_data)
                            st.session_state.current_embeddings_name = selected_set
                            st.success(f"✅ Loaded: {selected_set}")
                            st.rerun()
            
            with col2:
                if st.button("Delete", use_container_width=True, key="delete_btn"):
                    try:
                        embeddings_file = f"embeddings/{selected_set}.pkl"
                        if os.path.exists(embeddings_file):
                            os.remove(embeddings_file)
                            st.success(f"✅ Deleted {selected_set}")
                            st.rerun()
                        else:
                            st.error(f"File not found: {embeddings_file}")
                    except Exception as e:
                        st.error(f"Delete failed: {e}")
        else:
            st.info("No knowledge bases found. Upload documents to get started.")
        
        st.divider()
        
        #--Current Status--
        st.markdown("### 📊 Current Status")
        if st.session_state.current_embeddings_name:
            st.info(f"**Active:** {st.session_state.current_embeddings_name}")
            st.info(f"**Chunks:** {len(st.session_state.vector_store.chunks_data):,}")
            
            #--Calculate total content size--
            total_chars = sum(len(chunk.get('content', '')) for chunk in st.session_state.vector_store.chunks_data)
            st.info(f"**Content:** {total_chars:,} characters")
        else:
            st.warning("No knowledge base loaded")

# functions.py
# functions.py

def querry():
    """Handles user query input, voice input, and initiates processing with a clean layout."""
    
    st.markdown("---")
    st.markdown("#### 💬 Chat with your documents")

    # --- Restore the clean text input and button layout using a form ---
    with st.form(key='query_form', clear_on_submit=True):
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            user_query = st.text_input(
                "Ask a question:",
                value=st.session_state.user_query_input,
                placeholder="e.g., What are the main findings?",
                label_visibility="collapsed",
                key="query_text_input"
            )
        with col2:
            submit_button = st.form_submit_button(label="Ask")

    # --- Place the microphone button cleanly below the form ---
    if st.button("🎤 Use Voice Input"):
        with st.spinner("Recording... Speak now. Recording will stop automatically."):
            try:
                audio_file_path = record_audio_from_mic()
                transcribed_text = transcribe_audio(audio_file_path)
                
                if transcribed_text:
                    # Update the session state and rerun to show text in the box
                    st.session_state.user_query_input = transcribed_text
                    st.rerun()
                else:
                    st.warning("Could not understand audio or no speech was detected.")
            except Exception as e:
                logger.error(f"Error during voice input processing: {e}")
                st.error("An error occurred with voice input. Please check mic permissions.")
    
    # --- Process the query on submission ---
    if submit_button and user_query:
        st.session_state.user_query_input = "" # Clear state after submission
        if not st.session_state.get('vector_store') or not st.session_state.vector_store.get_all_chunks():
            st.warning("Please process documents before asking a question.")
            return

        with st.spinner("Thinking..."):
            try:
                st.session_state.messages.append({"role": "user", "content": user_query})
                answer, sources, rag_metrics = generate_structured_answer_with_gemini(
                    user_query, st.session_state.vector_store
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer, "sources": sources, "metrics": rag_metrics
                })
                st.session_state.chat_history.extend([
                    {"role": "user", "parts": [user_query]},
                    {"role": "model", "parts": [answer]}
                ])
                st.rerun() # Rerun to display the new message
            except Exception as e:
                logger.error(f"Error generating answer: {e}")
                st.error(f"An error occurred while generating the answer: {e}")
                
def chat_history():
    """Render stored chat messages with proper formatting and optional audio response."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Render oldest first
    for msg in reversed(st.session_state.chat_history):
        if msg["role"] == "user":
            with st.chat_message("user", avatar="🧑"):
                st.markdown(msg["content"])

        elif msg["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                # Assistant text
                st.markdown(msg["content"])

                # Show truncation warning if present
                if "truncated_notice" in msg:
                    st.info(msg["truncated_notice"], icon="⚠️")

                # FIXED: Handle both TTS modes for historical messages
                if msg.get("audio_id"):
                    # JavaScript TTS - recreate player
                    st.markdown("### 🔊 Audio Response")
                    try:
                        # Get the original text and settings for replay
                        clean_text = re.sub(r'[*_#>`\[\]()]', '', msg["content"])
                        clean_text = re.sub(r'http[s]?://[^\s]+', '', clean_text)
                        clean_text = re.sub(r'\s+', ' ', clean_text).strip()[:800]
                        
                        create_js_tts_player(
                            clean_text,
                            language=st.session_state.get("audio_language", "en"),
                            slow=st.session_state.get("audio_speed", False),
                            speech_id=msg["audio_id"],
                            auto_play=False
                        )
                    except Exception as e:
                        st.error(f"JavaScript TTS replay failed: {e}")
                
                elif msg.get("audio_file"):
                    # gTTS - recreate audio player if file still exists
                    st.markdown("### 🔊 Audio Response")
                    if os.path.exists(msg["audio_file"]):
                        try:
                            create_gtts_audio_player(msg["audio_file"], autoplay=False)
                        except Exception as e:
                            st.error(f"Google TTS replay failed: {e}")
                    else:
                        st.info("Audio file no longer available (automatically cleaned up)")

    # Chat Management 
    st.markdown('<h5 style="font-size: 1rem;">Chat Management</h5>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_processed_query = ""
            st.session_state.query_counter = 0
            st.success("✅ Chat history cleared!")
            st.rerun()

    with col2:
        if st.button("Export Chat", use_container_width=True):
            if st.session_state.get('chat_history'):
                chat_export = ""
                # Process in reverse to show chronological order
                for msg in reversed(st.session_state.chat_history):
                    role = "User" if msg["role"] == "user" else "Assistant"
                    chat_export += f"**{role}:** {msg['content']}\n\n"
                
                timestamp = int(time.time())
                st.download_button(
                    label="Download Chat",
                    data=chat_export,
                    file_name=f"chat_export_{timestamp}.txt",
                    mime="text/plain"
                )
            else:
                st.info("No chat history to export")

    with col3:
        chat_count = len([msg for msg in st.session_state.get('chat_history', []) if msg["role"] == "user"])
        st.markdown(f'<div class="questions-asked-box"><strong>Questions Asked</strong><br><span style="font-size: 1.5rem;">{chat_count}</span></div>', unsafe_allow_html=True)
    st.divider()

def tts_settings():
    with st.expander("🔊 Text-to-Speech Settings", expanded=False):
        st.markdown("#### Audio Response Settings")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.session_state.tts_enabled = st.checkbox(
                "Enable Audio Responses", 
                value=st.session_state.tts_enabled,
                help="Automatically generate speech for AI responses using JavaScript TTS")
        
        with col2:
            if st.session_state.tts_enabled:
                st.session_state.audio_speed = st.checkbox(
                    "Slow Speech",
                    value=st.session_state.get('audio_speed', False),
                    help="Enable slower speech rate for better comprehension"
                )
        
        with col3:
            # Test TTS button
            if st.button("🧪 Test TTS", help="Test if TTS is working in your browser"):
                test_text = "Hello! This is a test of the text-to-speech system. If you can hear this, TTS is working correctly."
                try:
                    speech_id = synthesize_speech(test_text, language='en', slow=False)
                    if speech_id:
                        st.success("✅ TTS test initiated - you should see audio controls above")
                    else:
                        st.error("❌ TTS test failed - check browser compatibility")
                except Exception as e:
                    st.error(f"❌ TTS Error: {e}")

#--System Info Interface--
def Systeminfo():
    with st.expander("ℹ️ System Information & Troubleshooting", expanded=False):
        st.markdown("### System Capabilities")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Available Features:**")
            features = [
                ("Speech Recognition", VOICE_FEATURES_AVAILABLE, "speech_recognition + Google API"),
                ("Text-to-Speech", TTS_AVAILABLE, "JavaScript Web Speech API (Browser)"),
                ("Audio Playback", AUDIO_PLAYBACK_AVAILABLE, "Browser Built-in"),
                ("AI Models", embedding_model is not None, "SentenceTransformer + CrossEncoder"),
                ("Document Processing", True, "PyPDF2 + Text processing")
            ]
            
            for feature, available, tech in features:
                status = "✅" if available else "⚠️"
                st.write(f"{status} **{feature}:** {tech}")
        
        with col2:
            st.markdown("**Installation Commands:**")
            st.code("""
            #Core RAG functionality
            pip install streamlit sentence-transformers
            pip install google-generativeai
            pip install PyPDF2 scikit-learn numpy

            # Voice features (minimal dependencies!)
            pip install speechrecognition
            pip install pyaudio  # For microphone input

            # Optional enhancements
            pip install rank-bm25 cross-encoder

            # TTS handled by browser - no extra dependencies!
            """, language="bash")
        st.markdown("### TTS Troubleshooting")
        
        troubleshooting_tips = [
            "**No audio output?** Click the Play button in the audio controls that appear after each response",
            "**Audio controls not appearing?** Try refreshing the page and ensure JavaScript is enabled",
            "**TTS not working?** Try Chrome/Firefox - these have the best Web Speech API support",
            "**Silent audio?** Check browser audio permissions and system volume",
            "**Microphone not working?** Install pyaudio: `pip install pyaudio`",
            "**Slow processing?** Try smaller document chunks or fewer documents",
            "**API errors?** Check your GEMINI_API_KEY environment variable",
            "**Import errors?** Reinstall dependencies from requirements.txt",
            "**Duplicate responses?** Fixed in this version - queries are now properly deduplicated"
        ]
        
        for tip in troubleshooting_tips:
            st.markdown(f"• {tip}")

#-Final Footer--
def Footer():
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 2rem;'>
        <h4>🚀 RAG Prototype</h4>
        <p><em>Empowering intelligent document interaction with browser-native voice synthesis</em></p>
    </div>
    """, unsafe_allow_html=True)
