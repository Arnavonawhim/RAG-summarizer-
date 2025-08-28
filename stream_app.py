
"""
RAG Prototype - Streamlit Application
This module provides a user-friendly web interface for the RAG pipeline
"""

import streamlit as st
import os
import logging

    # =============================================================================
    # CONFIGURATION AND INITIALIZATION
    # =============================================================================

# Configure logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

    # --- SESSION STATE INITIALIZATION ---
if 'embedding_model_loaded' not in st.session_state:
    st.session_state.embedding_model_loaded = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_query_input' not in st.session_state:
    st.session_state.user_query_input = ""
if 'tts_enabled' not in st.session_state:
    st.session_state.tts_enabled = True
if 'current_tts_mode' not in st.session_state:
    st.session_state.current_tts_mode = 'js'
if 'current_embeddings_name' not in st.session_state:
    st.session_state.current_embeddings_name = None

    # =============================================================================
    # IMPORTS
    # =============================================================================

    try:
        from functions import (css, config, main_title, sidebar, querry, chat_history, tts_settings, Systeminfo, Footer)
        from voice_utils import TTS_AVAILABLE
        logger.info("Successfully imported functions and voice utilities")
    except ImportError as e:
        logger.error(f"Failed to import a required module: {e}")
        st.error(f"A critical component is missing. Please check the logs. Error: {e}")
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
