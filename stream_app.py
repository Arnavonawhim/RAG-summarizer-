"""
RAG Prototype - Streamlit Application

This module provides a user-friendly web interface for the RAG pipeline
"""

import streamlit as st
import os
import re
import uuid
import logging
import tempfile
import time
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

try:
    from functions import(css,config,main_title,sidebar,querry,chat_history,tts_settings,Systeminfo,Footer)
    logger.info("Successfully imported functions")
except ImportError as e:
    logger.error(f"Failed to import functions: {e}")
    st.error(f"Required functions missing: {e}")
    st.stop()

# Import voice utilities (now with JavaScript TTS)
try:
    from voice_utils import ( create_voice_settings_interface,TTS_AVAILABLE)
    logger.info("Successfully imported voice utilities with JavaScript TTS")
    VOICE_FEATURES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Voice utilities not available: {e}")
    TTS_AVAILABLE = False
    AUDIO_PLAYBACK_AVAILABLE = False
    VOICE_FEATURES_AVAILABLE = False

# Google AI Configuration
import google.generativeai as genai
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
if not GEMINI_API_KEY:
    st.error("Google Gemini API Key not found in environment variables.")
    st.stop()
try:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Google Generative AI configured successfully")
except Exception as e:
    logger.error(f"Error configuring Google Generative AI: {e}")
    st.error(f"Error configuring Google Generative AI: {e}")
    st.stop()


#--page configuration--
config()

#--CSS Styling--
css()

#--Main Title and Feature Highlights--
main_title()

# Sidebar - Document Processing
sidebar()

querry()
# Display chat history 
chat_history()

# FIXED: Text-to-Speech Settings with Better Error Handling
if TTS_AVAILABLE:
    tts_settings()
# Advanced Audio Settings Panel (unchanged but improved)
    if VOICE_FEATURES_AVAILABLE:
        with st.expander("🎛️ Advanced Audio Settings", expanded=False):
            create_voice_settings_interface()

# System Information & Troubleshooting
Systeminfo()

# Footer
Footer()
