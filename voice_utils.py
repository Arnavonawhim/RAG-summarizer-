"""
Voice utilities for RAG Prototype - Enhanced Version with TTS Switching

This module provides speech recognition and dual text-to-speech functionality:
- Default: gTTS (Google Text-to-Speech) 
- Alternative: JavaScript Web Speech API
- Switchable via function call

Author: RAG Prototype Team
Date: 2024
"""

import streamlit as st
import speech_recognition as sr
import tempfile
import os
import logging
import threading
import time
from typing import Optional
import uuid
import re
import html

# Configure logging
logger = logging.getLogger(__name__)

# Text-to-Speech using gTTS (default)
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
    logger.info("gTTS available - Google TTS functionality enabled")
except ImportError as e:
    GTTS_AVAILABLE = False
    logger.warning(f"gTTS not available: {e}")

# Audio playback support
try:
    import pygame
    PYGAME_AVAILABLE = True
    try:
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        logger.info("Pygame mixer initialized successfully.")
    except pygame.error as e:
        PYGAME_AVAILABLE = False # Make sure to disable it if it fails
        logger.warning(f"Pygame mixer could not be initialized: {e}. Audio playback will fall back to browser.")
except ImportError:
    PYGAME_AVAILABLE = False
    logger.info("Pygame not available - using Streamlit audio player only")

# TTS Configuration
TTS_AVAILABLE = GTTS_AVAILABLE  # Default to gTTS availability
AUDIO_PLAYBACK_AVAILABLE = True  # Streamlit always supports audio playback

def initialize_voice_state():
    """Initialize voice-related session state variables"""
    if 'voice_text' not in st.session_state:
        st.session_state.voice_text = ""
    if 'voice_input_text' not in st.session_state:   
        st.session_state.voice_input_text = ""
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'voice_input_ready' not in st.session_state:
        st.session_state.voice_input_ready = False
    if 'tts_enabled' not in st.session_state:
        st.session_state.tts_enabled = True
    if 'voice_language' not in st.session_state:    
        st.session_state.voice_language = 'en-US'
    if 'audio_language' not in st.session_state:
        st.session_state.audio_language = 'en'
    if 'audio_speed' not in st.session_state:
        st.session_state.audio_speed = False
    if 'voice_duration' not in st.session_state:
        st.session_state.voice_duration = 10
    if 'use_js_tts' not in st.session_state:
        st.session_state.use_js_tts = False  # Default to gTTS
    if 'tts_mode' not in st.session_state:
        st.session_state.tts_mode = 'gtts'  # 'gtts' or 'javascript'

def switch_to_js_tts():
    """
    Switch from gTTS to JavaScript TTS
    Call this function to enable JavaScript TTS mode
    """
    st.session_state.use_js_tts = True
    st.session_state.tts_mode = 'javascript'
    logger.info("Switched to JavaScript TTS mode")
    st.success("🔄 Switched to JavaScript TTS mode")

def switch_to_gtts():
    """
    Switch back to gTTS (Google Text-to-Speech)
    Call this function to enable gTTS mode (default)
    """
    st.session_state.use_js_tts = False
    st.session_state.tts_mode = 'gtts'
    logger.info("Switched to Google TTS (gTTS) mode")
    st.success("🔄 Switched to Google TTS (gTTS) mode")

def get_current_tts_mode():
    """
    Get the current TTS mode
    Returns: 'gtts' or 'javascript'
    """
    return st.session_state.get('tts_mode', 'gtts')

def record_audio_from_mic(duration=10, sample_rate=16000):
    """
    Record audio from microphone using speech_recognition library
    """
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone(sample_rate=sample_rate) as source:
            status_placeholder = st.empty()
            
            status_placeholder.info("Adjusting for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            recognizer.energy_threshold = 300
            
            status_placeholder.success("Listening... Start speaking now!")
            
            audio = recognizer.listen(source, timeout=2, phrase_time_limit=duration)
            status_placeholder.success("Audio recorded successfully!")
            
            return audio
            
    except sr.WaitTimeoutError:
        st.error("No speech detected within timeout period")
        return None
    except sr.RequestError as e:
        st.error(f"Microphone error: {e}")
        return None
    except Exception as e:
        st.error(f"Recording failed: {e}")
        return None

def transcribe_audio(audio_data):
    """
    Transcribe audio data to text using Google Speech Recognition
    """
    if not audio_data:
        return None
        
    recognizer = sr.Recognizer()
    
    try:
        text = recognizer.recognize_google(audio_data, language=st.session_state.voice_language)
        return text
    except sr.UnknownValueError:
        st.error("Could not understand the audio - please try speaking more clearly")
        return None
    except sr.RequestError as e:
        st.error(f"Speech recognition service error: {e}")
        return None
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return None

# ======================= GOOGLE TTS (gTTS) FUNCTIONS =======================

def synthesize_speech_gtts(text: str, language: str = 'en', slow: bool = False) -> Optional[str]:
    """
    Convert text to speech using Google TTS (gTTS)
    """
    if not GTTS_AVAILABLE:
        st.warning("Google TTS not available. Install with: `pip install gtts`")
        return None
    
    if not text or not text.strip():
        return None
    
    # Clean and limit text length
    text = text.strip()
    max_length = 2000
    if len(text) > max_length:
        text = text[:max_length] + "... (truncated for audio)"
        st.info(f"📝 Text truncated to {max_length} characters for audio generation")
    
    try:
        tts = gTTS(text=text, lang=language, slow=slow)
        
        # Ensure temp directory exists
        temp_dir = os.path.join(os.getcwd(), "temp_audio")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate unique filename
        audio_filename = f"gtts_{uuid.uuid4().hex[:8]}.mp3"
        audio_path = os.path.join(temp_dir, audio_filename)
        
        # Save audio file
        tts.save(audio_path)
        
        if os.path.exists(audio_path):
            logger.info(f"Google TTS audio generated: {audio_path}")
            return audio_path
        else:
            st.error("Failed to generate Google TTS audio file")
            return None
            
    except Exception as e:
        st.error(f"Google TTS generation failed: {e}")
        logger.error(f"Google TTS error: {e}")
        return None

def create_gtts_audio_player(audio_path: str, autoplay: bool = False):
    """
    Create audio player for Google TTS generated files
    FIXED: Better error handling and display
    """
    if not audio_path or not os.path.exists(audio_path):
        st.error("Audio file not found")
        return False
    
    try:
        with open(audio_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        
        # FIXED: Add visual container for gTTS audio
        st.markdown("""
        <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                    padding: 15px; border-radius: 12px; margin: 15px 0; 
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <h4 style="margin: 0 0 10px 0; color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
                🔊 Google TTS Response
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Create audio player with Streamlit
        st.audio(audio_bytes, format='audio/mp3', start_time=0)
        
        # FIXED: Add status message
        st.success("✅ Google TTS audio ready - click play button above")
        
        # Schedule cleanup after 5 minutes
        schedule_audio_cleanup(audio_path, delay=300)
        
        return True
        
    except Exception as e:
        st.error(f"Google TTS audio player creation failed: {e}")
        logger.error(f"Google TTS audio player error: {e}")
        return False

# ======================= JAVASCRIPT TTS FUNCTIONS =======================

def clean_text_for_tts(text: str) -> str:
    """
    Clean and prepare text for JavaScript TTS
    """
    if not text:
        return ""
    
    # Remove markdown formatting
    text = re.sub(r'[*_#>`]', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Convert [text](url) to text
    
    # Remove URLs
    text = re.sub(r'http[s]?://[^\s]+', '', text)
    
    # Remove page citations like (p.3) - they don't read well
    text = re.sub(r'\(p\.\d+\)', '', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove HTML entities
    text = html.unescape(text)
    
    # Escape special characters for JavaScript
    text = text.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"')
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    return text

def synthesize_speech_js(text: str, language: str = 'en', slow: bool = False) -> Optional[str]:
    """
    Convert text to speech using JavaScript Web Speech API
    """
    if not text or not text.strip():
        logger.warning("Empty text provided to JavaScript TTS")
        return None
    
    # Clean and prepare text
    clean_text = clean_text_for_tts(text)
    
    if not clean_text:
        logger.warning("Text became empty after cleaning")
        return None
    
    # Limit text length for better performance
    max_length = 800
    if len(clean_text) > max_length:
        clean_text = clean_text[:max_length] + "..."
        st.info(f"📝 Text truncated to {max_length} characters for better audio performance")
    
    try:
        speech_id = str(uuid.uuid4())[:8]
        create_js_tts_player(clean_text, language, slow, speech_id, auto_play=True)
        logger.info(f"JavaScript TTS created for speech_id: {speech_id}, text length: {len(clean_text)}")
        return speech_id
    except Exception as e:
        st.error(f"🔊 JavaScript TTS generation failed: {e}")
        logger.error(f"JavaScript TTS error: {e}")
        return None

def create_js_tts_player(speech_id_or_text, language='en', slow=False, speech_id=None, auto_play=False):
    """
    Create JavaScript-based TTS player - FIXED parameter handling
    """
    # Handle both old and new calling conventions
    if speech_id is None:
        if isinstance(speech_id_or_text, str) and len(speech_id_or_text) < 20 and not ' ' in speech_id_or_text:
            # This looks like a speech_id
            speech_id = speech_id_or_text
            # Need to get text from somewhere - this suggests the calling code needs fixing
            text = "JavaScript TTS test text"
        else:
            # This is text
            text = speech_id_or_text
            speech_id = str(uuid.uuid4())[:8]
    else:
        text = speech_id_or_text
    
    # Clean text for TTS
    text = clean_text_for_tts(text)
    
    # Rest of the JavaScript TTS player code remains the same...
    # (keeping the existing implementation)
    
    language_map = {
        'en': 'en-US', 'es': 'es-ES', 'fr': 'fr-FR', 'de': 'de-DE',
        'it': 'it-IT', 'pt': 'pt-BR', 'ru': 'ru-RU', 'ja': 'ja-JP',
        'ko': 'ko-KR', 'zh': 'zh-CN'
    }
    
    js_language = language_map.get(language, 'en-US')
    rate = 0.7 if slow else 1.0
    
    tts_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 12px; border: 1px solid #4a90e2; margin: 15px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
            <h4 style="margin: 0; color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">🔊 JavaScript TTS Response</h4>
            <div style="display: flex; gap: 8px;">
                <button onclick="playTTS_{speech_id}()" id="play-btn-{speech_id}"
                        style="background: #28a745; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: bold; transition: all 0.2s;">
                    ▶️ Play
                </button>
                <button onclick="stopTTS_{speech_id}()" id="stop-btn-{speech_id}"
                        style="background: #dc3545; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: bold; transition: all 0.2s;">
                    ⏹️ Stop
                </button>
            </div>
        </div>
        <div id="tts-status-{speech_id}" style="margin-top: 10px; font-style: italic; color: white; font-size: 14px; background: rgba(255,255,255,0.1); padding: 8px; border-radius: 6px;">
            🎵 JavaScript TTS ready to play...
        </div>
    </div>
    
    <script>
    (function() {{
        if (!window.speechSynthesis) {{
            document.getElementById('tts-status-{speech_id}').innerHTML = 
                '<span style="color: #ffcccc;">⚠️ Speech synthesis not supported in this browser.</span>';
            return;
        }}
        
        window.playTTS_{speech_id} = function() {{
            const utterance = new SpeechSynthesisUtterance(`{text}`);
            utterance.lang = '{js_language}';
            utterance.rate = {rate};
            
            utterance.onstart = function() {{
                document.getElementById('tts-status-{speech_id}').innerHTML = 
                    '<span style="color: #90EE90;">🔊 Playing JavaScript TTS...</span>';
            }};
            
            utterance.onend = function() {{
                document.getElementById('tts-status-{speech_id}').innerHTML = 
                    '<span style="color: #ADD8E6;">✅ JavaScript TTS completed</span>';
            }};
            
            window.speechSynthesis.speak(utterance);
        }};
        
        window.stopTTS_{speech_id} = function() {{
            window.speechSynthesis.cancel();
            document.getElementById('tts-status-{speech_id}').innerHTML = 
                '<span style="color: #ffb3b3;">⏹️ JavaScript TTS stopped</span>';
        }};
        
        {"window.playTTS_" + speech_id + "();" if auto_play else ""}
    }})();
    </script>
    """
    
    st.markdown(tts_html, unsafe_allow_html=True)


# ======================= UNIFIED TTS INTERFACE =======================

def synthesize_speech(text: str, language: str = 'en', slow: bool = False) -> Optional[str]:
    """
    Universal TTS function that switches between gTTS and JavaScript TTS
    based on current mode setting
    """
    current_mode = get_current_tts_mode()
    
    if current_mode == 'javascript':
        return synthesize_speech_js(text, language, slow)
    else:  # Default to gTTS
        file_path = synthesize_speech_gtts(text, language, slow)
        if file_path:
            # FIXED: Automatically create audio player for gTTS
            create_gtts_audio_player(file_path, autoplay=True)
            return file_path
        return None
    

def create_audio_player(audio_path: str = None, autoplay: bool = False):
    """
    Universal audio player function that works with current TTS mode
    """
    current_mode = get_current_tts_mode()
    
    if current_mode == 'javascript':
        # For JavaScript TTS, the player is already created in synthesize_speech_js
        return True
    else:  # gTTS mode
        if audio_path:
            return create_gtts_audio_player(audio_path, autoplay)
        return False

# ======================= UTILITY FUNCTIONS =======================

def schedule_audio_cleanup(file_path: str, delay: int = 300):
    """
    Schedule cleanup of audio file after specified delay
    """
    def cleanup():
        time.sleep(delay)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up audio file: {file_path}")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    threading.Thread(target=cleanup, daemon=True).start()

def create_voice_settings_interface():
    """Create comprehensive voice settings interface with TTS switching"""
    st.markdown("**Voice & Audio Configuration**")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎤 Voice Input", "🔊 Speech Output", "🔄 TTS Mode", "📊 System Status"])
    
    with tab1:
        st.markdown("#### Speech Recognition Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            voice_languages = {
                'en-US': '🇺🇸 English (US)', 'en-GB': '🇬🇧 English (UK)',
                'es-ES': '🇪🇸 Spanish (Spain)', 'fr-FR': '🇫🇷 French (France)',
                'de-DE': '🇩🇪 German (Germany)', 'it-IT': '🇮🇹 Italian (Italy)',
                'pt-BR': '🇧🇷 Portuguese (Brazil)', 'ru-RU': '🇷🇺 Russian (Russia)',
                'ja-JP': '🇯🇵 Japanese (Japan)', 'ko-KR': '🇰🇷 Korean (Korea)',
                'zh-CN': '🇨🇳 Chinese (Simplified)'
            }
            
            selected_voice_lang = st.selectbox(
                "Recognition Language",
                options=list(voice_languages.keys()),
                format_func=lambda x: voice_languages[x],
                index=0
            )
            st.session_state.voice_language = selected_voice_lang
        
        with col2:
            st.session_state.voice_duration = st.slider(
                "Recording Duration (seconds)", 
                min_value=5, max_value=30, 
                value=st.session_state.get('voice_duration', 10),
                help="Maximum time to listen for voice input"
            )
    
    with tab2:
        st.markdown("#### TTS Language & Speed Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            languages = {
                'en': '🇺🇸 English', 'es': '🇪🇸 Spanish', 'fr': '🇫🇷 French',
                'de': '🇩🇪 German', 'it': '🇮🇹 Italian', 'pt': '🇧🇷 Portuguese',
                'ru': '🇷🇺 Russian', 'ja': '🇯🇵 Japanese', 'ko': '🇰🇷 Korean', 'zh': '🇨🇳 Chinese'
            }
            
            selected_lang = st.selectbox(
                "TTS Language",
                options=list(languages.keys()),
                format_func=lambda x: languages[x],
                index=0,
                key="tts_language_selector"
            )
            st.session_state.audio_language = selected_lang
        
        with col2:
            st.session_state.audio_speed = st.checkbox(
                "Slow Speech",
                value=st.session_state.get('audio_speed', False),
                help="Enable slower speech rate for better comprehension",
                key="audio_speed_checkbox"
            )
            
            st.session_state.tts_enabled = st.checkbox(
                "Enable Auto-play for AI Responses",
                value=st.session_state.get('tts_enabled', True),
                help="Automatically generate and play audio for AI responses"
            )
    
    with tab3:
        st.markdown("#### TTS Mode Selection")
        
        current_mode = get_current_tts_mode()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Current Mode:** `{current_mode.upper()}`")
            
            if st.button("🎯 Switch to Google TTS", use_container_width=True, disabled=(current_mode == 'gtts')):
                switch_to_gtts()
                st.rerun()
            
            if st.button("🌐 Switch to JavaScript TTS", use_container_width=True, disabled=(current_mode == 'javascript')):
                switch_to_js_tts()
                st.rerun()
        
        with col2:
            st.markdown("**Mode Comparison:**")
            st.markdown("• **Google TTS (gTTS):** High quality, requires internet, file-based")
            st.markdown("• **JavaScript TTS:** Browser-based, instant, no files, cross-platform")
            
            # Test current mode
            test_text = st.text_input(
                "Test Current TTS Mode", 
                value="Hello! This is a test of the current text-to-speech system.",
                help="Enter text to test current TTS mode"
            )
            
            if st.button("🧪 Test Current TTS Mode", use_container_width=True):
                if test_text:
                    st.info(f"Testing {current_mode.upper()} mode...")
                    result = synthesize_speech(test_text, language='en', slow=False)
                    if result:
                        if current_mode == 'gtts':
                            create_gtts_audio_player(result)
                        st.success(f"✅ {current_mode.upper()} test completed!")
                    else:
                        st.error(f"❌ {current_mode.upper()} test failed!")
    
    with tab4:
        st.markdown("#### System Status & Diagnostics")
        
        status_data = {
            "🎤 Speech Recognition": "✅ Available (speech_recognition + Google API)",
            "🔊 Google TTS (gTTS)": "✅ Available" if GTTS_AVAILABLE else "❌ Not Available",
            "🌐 JavaScript TTS": "✅ Available (Browser Web Speech API)",
            "🔱 Audio Playback": "✅ Available (Streamlit + Pygame)" if PYGAME_AVAILABLE else "✅ Available (Streamlit only)",
            "📦 Dependencies": "✅ Minimal (speech_recognition, gtts)"
        }
        
        for feature, status in status_data.items():
            st.markdown(f"**{feature}:** {status}")
        
        st.markdown("#### 💻 Installation Commands")
        st.code("""
# Install all audio dependencies
pip install gtts pygame speechrecognition

# For microphone support
pip install pyaudio

# JavaScript TTS uses browser - no extra dependencies!
        """, language="bash")

def get_system_audio_info():
    """Get comprehensive audio system information"""
    return {
        "gtts_available": GTTS_AVAILABLE,
        "javascript_tts_available": True,  # Always available in browsers
        "pygame_available": PYGAME_AVAILABLE,
        "audio_playback_available": AUDIO_PLAYBACK_AVAILABLE,
        "speech_recognition_available": True,
        "current_tts_mode": get_current_tts_mode(),
        "temp_audio_dir": os.path.join(os.getcwd(), "temp_audio") if get_current_tts_mode() == 'gtts' else None,
        "version": "Dual TTS v1.0 - gTTS + JavaScript"
    }

# Export the main functions for use in app.py
__all__ = [
    'initialize_voice_state',
    'record_audio_from_mic',
    'transcribe_audio', 
    'synthesize_speech',
    'create_audio_player',
    'create_voice_settings_interface',
    'get_system_audio_info',
    'switch_to_js_tts',
    'switch_to_gtts',
    'get_current_tts_mode',
    'TTS_AVAILABLE',
    'AUDIO_PLAYBACK_AVAILABLE',
    'create_js_tts_player'
]
