"""
Text Reader App - Refactored with LangGraph Agents
Multi-agent system for file processing, text extraction, grammar checking, and TTS conversion
"""

import streamlit as st
import os
import base64
from dotenv import load_dotenv
from agents import (
    create_text_reader_workflow, 
    TextReaderState, 
    get_api_usage_info, 
    check_api_rate_limit, 
    estimate_requests_needed,
    GrammarAnalysisAgent,
    ParallelEditingWindowAgent,
    HumanEditingAgent
)
from parallel_editor_ui import ParallelEditorUI

# Load environment variables
load_dotenv()

# Try to import EasyOCR, handle gracefully if not available
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Text Reader App - LangGraph",
    page_icon="📖",
    layout="wide"
)

st.title("Text Reader App - Multi-Agent System")
st.markdown("*Powered by LangGraph agents for file processing, text extraction, grammar checking, and TTS conversion*")

def get_audio_download_link(audio_bytes, filename="speech.mp3"):
    """Generate download link for audio file"""
    b64 = base64.b64encode(audio_bytes).decode()
    href = f'<a href="data:audio/mp3;base64,{b64}" download="{filename}">Download Audio File</a>'
    return href

def get_base_filename_from_state():
    """Extract the first uploaded filename without extension for use as base export name"""
    try:
        current_state = st.session_state.get('current_state', {})
        extracted_texts = current_state.get("extracted_texts", [])
        if extracted_texts:
            filename = extracted_texts[0].get("filename", "")
            if filename:
                # Remove extension
                import os
                base_name = os.path.splitext(filename)[0]
                return base_name
        return None
    except Exception:
        return None

def initialize_session_state():
    """Initialize session state variables"""
    if 'workflow_app' not in st.session_state:
        workflow_app, agents = create_text_reader_workflow()
        st.session_state.workflow_app = workflow_app
        st.session_state.agents = agents
    
    # Initialize grammar analysis agents
    if 'grammar_analyzer' not in st.session_state:
        st.session_state.grammar_analyzer = GrammarAnalysisAgent()
    
    if 'parallel_editor' not in st.session_state:
        st.session_state.parallel_editor = ParallelEditingWindowAgent()
    
    if 'parallel_editor_ui' not in st.session_state:
        st.session_state.parallel_editor_ui = ParallelEditorUI()
    
    if 'current_state' not in st.session_state:
        st.session_state.current_state = {
            "extracted_texts": [],
            "processed_chunks": [],
            "audio_files": [],
            "spell_checked": False,
            "grammar_checked": False,
            "human_edited": False,
            "editing_mode": False,
            "current_step": "initial",
            "error_message": None
        }
    else:
        # Ensure existing state has all required keys
        required_keys = {
            "extracted_texts": [],
            "processed_chunks": [],
            "audio_files": [],
            "spell_checked": False,
            "grammar_checked": False,
            "human_edited": False,
            "editing_mode": False,
            "current_step": "initial",
            "error_message": None
        }
        for key, default_value in required_keys.items():
            if key not in st.session_state.current_state:
                st.session_state.current_state[key] = default_value
    
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

# Initialize session state
initialize_session_state()

# Sidebar for settings
st.sidebar.header("⚙️ Agent Settings")

# AI Provider Selection
ai_provider = st.sidebar.selectbox(
    "Default AI Provider",
    options=["OpenAI", "Ollama"],
    help="Choose between OpenAI GPT models or local Ollama models for text processing"
)

# Advanced AI Provider Options
with st.sidebar.expander("🔧 Advanced AI Settings", expanded=False):
    spell_check_ai = st.selectbox(
        "Spell Check AI Provider",
        options=["Use Default", "OpenAI", "Ollama"],
        help="Choose AI provider specifically for spell checking"
    )
    
    summarization_ai = st.selectbox(
        "Summarization AI Provider", 
        options=["Use Default", "OpenAI", "Ollama"],
        help="Choose AI provider specifically for text summarization"
    )
    
    # Show helpful info for Ollama summarization
    if summarization_ai == "Ollama" or (summarization_ai == "Use Default" and ai_provider == "Ollama"):
        st.info("📋 **Ollama Summarization**: Uses your local Ollama server for text summarization. "
                "Make sure Ollama is running and configure the URL/model below.")

# TTS Engine Selection
tts_engine = st.sidebar.selectbox(
    "TTS Engine",
    options=["Google TTS (Free)", "OpenAI TTS (Premium)", "Mozilla TTS (Local)", "Coqui TTS (Local)"],
    help="Choose between free Google TTS, premium OpenAI TTS, local Mozilla TTS, or advanced Coqui TTS"
)

# AI Provider Settings
openai_api_key = None
ollama_url = None
ollama_model = None

if ai_provider == "OpenAI":
    # Try to load from environment first
    env_api_key = os.getenv("OPENAI_API_KEY")
    
    if env_api_key:
        # Use environment variable, don't show input field
        openai_api_key = env_api_key
        st.sidebar.success("✅ Using OpenAI API key from .env file")
    else:
        # Show input field only if no .env key exists
        openai_api_key = st.sidebar.text_input(
            "OpenAI API Key", 
            type="password",
            help="Enter your OpenAI API key for text processing and premium TTS"
        )
# Show Ollama settings if any AI provider uses Ollama
if ai_provider == "Ollama" or spell_check_ai == "Ollama" or summarization_ai == "Ollama":
    ollama_url = st.sidebar.text_input(
        "Ollama URL",
        value="http://localhost:11434",
        help="URL of your Ollama server (default: http://localhost:11434)"
    )
    
    ollama_model = st.sidebar.text_input(
        "Ollama Model",
        value="llama2",
        help="Ollama model name (e.g., llama2, mistral, codellama)"
    )
    
    if st.sidebar.button("🔍 Test Ollama Connection"):
        with st.sidebar:
            with st.spinner("Testing Ollama connection..."):
                try:
                    import requests
                    response = requests.get(f"{ollama_url}/api/tags", timeout=5)
                    if response.status_code == 200:
                        models = response.json().get('models', [])
                        model_names = [m['name'] for m in models]
                        st.success(f"✅ Connected! Available models: {', '.join(model_names)}")
                    else:
                        st.error(f"❌ Connection failed (status: {response.status_code})")
                except Exception as e:
                    st.error(f"❌ Connection failed: {str(e)}")
else:
    # Set default values when Ollama is not selected
    ollama_url = None
    ollama_model = None

# Resolve AI provider settings
actual_spell_check_ai = ai_provider if spell_check_ai == "Use Default" else spell_check_ai
actual_summarization_ai = ai_provider if summarization_ai == "Use Default" else summarization_ai

# OpenAI TTS Settings (only if using OpenAI TTS)
if tts_engine == "OpenAI TTS (Premium)":
    if ai_provider != "OpenAI":
        st.sidebar.warning("⚠️ OpenAI TTS requires OpenAI as AI provider for API key")
    
    openai_voice = st.sidebar.selectbox(
        "OpenAI Voice",
        options=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        help="Choose OpenAI voice style"
    )
    
    openai_model = st.sidebar.selectbox(
        "OpenAI Model",
        options=["tts-1", "tts-1-hd"],
        index=1,
        help="tts-1-hd for higher quality (costs more)"
    )
elif tts_engine == "Mozilla TTS (Local)":
    # Mozilla TTS Settings
    mozilla_model = st.sidebar.selectbox(
        "Mozilla TTS Model",
        options=["tts_models/en/ljspeech/tacotron2-DDC", "tts_models/en/ljspeech/fast_speech", 
                "tts_models/en/ljspeech/speedy_speech", "tts_models/en/sam/tacotron-DDC"],
        help="Choose Mozilla TTS model (requires model to be installed)"
    )
    
    mozilla_vocoder = st.sidebar.selectbox(
        "Mozilla TTS Vocoder",
        options=["vocoder_models/en/ljspeech/hifigan_v2", "vocoder_models/en/ljspeech/multiband-melgan",
                "vocoder_models/universal/libri-tts/wavegrad", "vocoder_models/en/ljspeech/univnet"],
        help="Choose Mozilla TTS vocoder for audio generation"
    )
    
    if st.sidebar.button("🔍 Test Mozilla TTS"):
        with st.sidebar:
            with st.spinner("Testing Mozilla TTS..."):
                try:
                    # Test Mozilla TTS availability
                    import TTS
                    from TTS.api import TTS as TTSApi
                    st.success("✅ Mozilla TTS is available!")
                except ImportError:
                    st.error("❌ Mozilla TTS not installed. Run: pip install TTS")
                except Exception as e:
                    st.error(f"❌ Mozilla TTS test failed: {str(e)}")
    
    # Show installation instructions for Mozilla TTS
    st.sidebar.info("💡 **Mozilla TTS Setup:**\n"
                   "1. Install: `pip install TTS`\n"
                   "2. First run will download models automatically\n"
                   "3. Models are stored locally (~1-2GB per model)")
elif tts_engine == "Coqui TTS (Local)":
    # Coqui TTS Settings
    coqui_model = st.sidebar.selectbox(
        "Coqui TTS Model",
        options=[
            "tts_models/multilingual/multi-dataset/xtts_v2",  # XTTS v2 - best quality
            "tts_models/en/ljspeech/tacotron2-DDC",
            "tts_models/en/ljspeech/glow-tts",
            "tts_models/en/ljspeech/speedy-speech",
            "tts_models/en/ljspeech/neural_hmm",
            "tts_models/en/vctk/vits",
            "tts_models/en/vctk/fast_pitch",
        ],
        help="Choose Coqui TTS model (XTTS v2 supports voice cloning and multilingual)"
    )
    
    # Voice cloning options for XTTS v2
    if "xtts_v2" in coqui_model:
        st.sidebar.subheader("🎭 Voice Cloning (XTTS v2)")
        
        # Speaker selection
        speaker_option = st.sidebar.radio(
            "Speaker Source",
            options=["Built-in Speaker", "Upload Audio Sample"],
            help="Choose between built-in voices or clone a voice from audio sample"
        )
        
        if speaker_option == "Built-in Speaker":
            coqui_speaker = st.sidebar.selectbox(
                "Built-in Speaker",
                options=["Claribel Dervla", "Daisy Studious", "Gracie Wise", "Tammie Ema", 
                        "Alison Dietlinde", "Ana Florence", "Annmarie Nele", "Asya Anara", 
                        "Brenda Stern", "Gitta Nikolina", "Henriette Usha", "Sofia Hellen"],
                help="Choose from high-quality built-in speaker voices"
            )
            coqui_reference_audio = None
        else:
            coqui_speaker = "custom"
            coqui_reference_audio = st.sidebar.file_uploader(
                "Upload Reference Audio (3-10 seconds)",
                type=['wav', 'mp3', 'flac'],
                help="Upload a short, clear audio sample of the voice you want to clone"
            )
        
        # Language selection for XTTS v2
        coqui_language = st.sidebar.selectbox(
            "Language",
            options=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja"],
            format_func=lambda x: {
                "en": "English", "es": "Spanish", "fr": "French", "de": "German", 
                "it": "Italian", "pt": "Portuguese", "pl": "Polish", "tr": "Turkish",
                "ru": "Russian", "nl": "Dutch", "cs": "Czech", "ar": "Arabic",
                "zh-cn": "Chinese", "ja": "Japanese"
            }[x],
            help="Choose language for speech generation"
        )
    else:
        # Standard TTS settings for non-XTTS models
        coqui_speaker = None
        coqui_reference_audio = None
        coqui_language = "en"
    
    if st.sidebar.button("🔍 Test Coqui TTS"):
        with st.sidebar:
            with st.spinner("Testing Coqui TTS..."):
                try:
                    # Test Coqui TTS availability
                    import TTS
                    from TTS.api import TTS as CoquiTTS
                    st.success("✅ Coqui TTS is available!")
                    
                    # Test model loading (this might take a moment)
                    try:
                        tts_test = CoquiTTS(model_name=coqui_model)
                        st.success(f"✅ Model '{coqui_model}' loaded successfully!")
                    except Exception as e:
                        st.warning(f"⚠️ Model loading test failed: {str(e)}")
                        st.info("Model will be downloaded on first use")
                        
                        # Provide specific troubleshooting
                        if "CUDA" in str(e):
                            st.info("💡 GPU not available, will use CPU (slower but works)")
                        elif "torch" in str(e):
                            st.error("PyTorch installation issue. Try: pip install torch")
                        elif "model" in str(e).lower():
                            st.info("💡 Model download needed. First TTS generation will be slow.")
                        
                except ImportError:
                    st.error("❌ Coqui TTS not installed. Run: pip install TTS")
                except Exception as e:
                    st.error(f"❌ Coqui TTS test failed: {str(e)}")
    
    # Show installation and feature instructions for Coqui TTS
    st.sidebar.info("💡 **Coqui TTS Features:**\n"
                   "• XTTS v2: Voice cloning & multilingual\n"
                   "• High-quality neural voices\n"
                   "• Custom speaker support\n"
                   "• Runs entirely offline")
    
    with st.sidebar.expander("🚀 Installation Guide", expanded=False):
        st.markdown("**Try these methods in order:**")
        st.code("# Method 1: Standard install\npip install TTS --upgrade", language="bash")
        st.code("# Method 2: From source\npip install git+https://github.com/coqui-ai/TTS.git", language="bash")
        st.code("# Method 3: With all dependencies\npip install TTS[all] --upgrade", language="bash")
        st.code("# Method 4: Using conda\nconda install -c conda-forge tts", language="bash")
        st.markdown("**Requirements:** Python 3.8-3.11, ~1.8GB for XTTS v2 model")
        
        st.markdown("**Windows users:** Install Microsoft C++ Build Tools first")
        st.markdown("**Linux users:** Install espeak and build essentials first")
        st.markdown("**Mac users:** Install Xcode command line tools first")
else:
    # Google TTS Settings
    language = st.sidebar.selectbox("Language", 
        options=['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh'],
        format_func=lambda x: {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 
            'de': 'German', 'it': 'Italian', 'pt': 'Portuguese',
            'ru': 'Russian', 'ja': 'Japanese', 'ko': 'Korean', 'zh': 'Chinese'
        }[x]
    )

# OCR Engine Selection
st.sidebar.header("OCR Settings")
available_engines = ["Tesseract"]
if EASYOCR_AVAILABLE:
    available_engines.append("EasyOCR")

ocr_engine = st.sidebar.selectbox(
    "OCR Engine for Images",
    options=available_engines,
    help="Choose OCR engine for extracting text from images. EasyOCR often works better for handwritten text and complex layouts."
)

if not EASYOCR_AVAILABLE and len(available_engines) == 1:
    st.sidebar.info("TIP: Install EasyOCR for better OCR accuracy: pip install easyocr")

# API Usage Information
st.sidebar.header("📊 API Usage Monitor")
usage_info = get_api_usage_info()

# Display current daily usage with color coding
usage_percentage = usage_info["usage_percentage"]
if usage_percentage < 50:
    st.sidebar.success(f"✅ **Daily Usage**: {usage_info['requests_used']:,}/{usage_info['daily_limit']:,} ({usage_percentage:.1f}%)")
elif usage_percentage < 80:
    st.sidebar.warning(f"⚠️ **Daily Usage**: {usage_info['requests_used']:,}/{usage_info['daily_limit']:,} ({usage_percentage:.1f}%)")
else:
    st.sidebar.error(f"🚨 **Daily Usage**: {usage_info['requests_used']:,}/{usage_info['daily_limit']:,} ({usage_percentage:.1f}%)")

st.sidebar.caption(f"Remaining: {usage_info['requests_remaining']:,} requests")
st.sidebar.caption(f"Date: {usage_info['date']}")

# Daily progress bar
st.sidebar.progress(min(usage_percentage / 100, 1.0), text="Daily Limit")

# Display per-minute token usage
minute_usage_percentage = usage_info.get("minute_usage_percentage", 0)
tokens_current_minute = usage_info.get("tokens_current_minute", 0)
tokens_per_minute_limit = usage_info.get("tokens_per_minute_limit", 20000)

st.sidebar.markdown("### 🕐 Per-Minute Token Usage")
if minute_usage_percentage < 50:
    st.sidebar.success(f"✅ **Current Minute**: {tokens_current_minute:,}/{tokens_per_minute_limit:,} tokens ({minute_usage_percentage:.1f}%)")
elif minute_usage_percentage < 80:
    st.sidebar.warning(f"⚠️ **Current Minute**: {tokens_current_minute:,}/{tokens_per_minute_limit:,} tokens ({minute_usage_percentage:.1f}%)")
else:
    st.sidebar.error(f"🚨 **Current Minute**: {tokens_current_minute:,}/{tokens_per_minute_limit:,} tokens ({minute_usage_percentage:.1f}%)")

# Per-minute progress bar
st.sidebar.progress(min(minute_usage_percentage / 100, 1.0), text="Per-Minute Limit")

# Total tokens used today
st.sidebar.caption(f"Total tokens today: {usage_info.get('tokens_used_today', 0):,}")

# Main content area
st.header("📁 Upload Files")
uploaded_files = st.file_uploader(
    "Choose files",
    type=['pdf', 'jpg', 'jpeg', 'png', 'gif'],
    help="Upload PDF, JPG, PNG, or GIF files",
    accept_multiple_files=True
)

if uploaded_files:
    # File ordering interface
    st.subheader("📋 File Order")
    
    # Initialize file order in session state
    if 'file_order' not in st.session_state or len(st.session_state.file_order) != len(uploaded_files):
        st.session_state.file_order = list(range(len(uploaded_files)))
    
    # Display current order with reordering controls
    for i, file_idx in enumerate(st.session_state.file_order):
        col1, col2, col3 = st.columns([4, 1, 1])
        with col1:
            st.write(f"{i+1}. {uploaded_files[file_idx].name}")
        with col2:
            if st.button("⬆️", key=f"up_{i}", disabled=(i == 0)):
                st.session_state.file_order[i], st.session_state.file_order[i-1] = st.session_state.file_order[i-1], st.session_state.file_order[i]
                st.rerun()
        with col3:
            if st.button("⬇️", key=f"down_{i}", disabled=(i == len(st.session_state.file_order)-1)):
                st.session_state.file_order[i], st.session_state.file_order[i+1] = st.session_state.file_order[i+1], st.session_state.file_order[i]
                st.rerun()
    
    # Show existing work summary if any
    if st.session_state.get("current_state") and st.session_state.current_state.get("extracted_texts"):
        existing_work = st.session_state.current_state
        total_texts = len(existing_work.get("extracted_texts", []))
        total_chunks = len(existing_work.get("processed_chunks", []))
        total_audio = len(existing_work.get("audio_files", []))
        
        st.info(f"📋 **Existing Work**: {total_texts} files processed, {total_chunks} text chunks, {total_audio} audio files")
        
        col1, col2 = st.columns(2)
        with col1:
            process_button = st.button("🚀 Process Files with Agents", type="primary")
        with col2:
            if st.button("🗑️ Clear All Work", help="Clear all previous work and start fresh"):
                st.session_state.current_state = {}
                st.session_state.processing_complete = False
                st.success("✅ All previous work cleared!")
                st.rerun()
    else:
        process_button = st.button("🚀 Process Files with Agents", type="primary")
    
    # Process files button
    if process_button:
        # Prepare TTS settings
        if tts_engine == "OpenAI TTS (Premium)":
            tts_settings = {
                "voice": openai_voice,
                "model": openai_model
            }
        elif tts_engine == "Mozilla TTS (Local)":
            tts_settings = {
                "model": mozilla_model,
                "vocoder": mozilla_vocoder
            }
        elif tts_engine == "Coqui TTS (Local)":
            tts_settings = {
                "model": coqui_model,
                "speaker": coqui_speaker,
                "language": coqui_language,
                "reference_audio": coqui_reference_audio.read() if coqui_reference_audio else None
            }
        else:
            tts_settings = {
                "language": language
            }
        
        # Get existing state (should already be properly initialized)
        existing_state = st.session_state.current_state
        
        # Validate state structure (should not be needed but safety check)
        required_keys = ["extracted_texts", "processed_chunks", "audio_files"]
        if not all(key in existing_state for key in required_keys):
            st.error("❌ Session state corruption detected! Reinitializing...")
            # Force reinitialization
            st.session_state.current_state = {
                "extracted_texts": [],
                "processed_chunks": [],
                "audio_files": [],
                "spell_checked": False,
                "grammar_checked": False,
                "human_edited": False,
                "editing_mode": False,
                "current_step": "initial",
                "error_message": None
            }
            existing_state = st.session_state.current_state
        
        # Preserve existing work while adding new files
        existing_extracted_texts = existing_state.get("extracted_texts", [])
        existing_processed_chunks = existing_state.get("processed_chunks", [])
        existing_audio_files = existing_state.get("audio_files", [])
        
        # Create initial state for the workflow
        initial_state = TextReaderState(
            uploaded_files=uploaded_files,
            file_order=st.session_state.file_order,
            extracted_texts=existing_extracted_texts,  # Preserve existing work
            processed_chunks=existing_processed_chunks,  # Preserve existing work
            spell_checked=existing_state.get("spell_checked", False),
            grammar_checked=existing_state.get("grammar_checked", False),
            human_edited=existing_state.get("human_edited", False),
            audio_files=existing_audio_files,  # Preserve existing work
            current_step="file_processing",
            error_message=None,
            api_key=openai_api_key,
            tts_engine=tts_engine,
            tts_settings=tts_settings,
            ocr_engine=ocr_engine,
            editing_mode=existing_state.get("editing_mode", False)
        )
        
        # Debug: Show initial state details
        st.info(f"🔧 Initial state created with {len(uploaded_files)} files, {len(existing_extracted_texts)} existing texts")
        st.info(f"🔧 File order: {st.session_state.file_order[:10]}{'...' if len(st.session_state.file_order) > 10 else ''}")
        
        # Debug: Validate uploaded files integrity
        st.info(f"🔍 File validation: {[f.name for f in uploaded_files]}")
        for i, file in enumerate(uploaded_files):
            if hasattr(file, 'size') and file.size > 0:
                st.success(f"✅ File {i}: {file.name} ({file.size} bytes)")
            else:
                st.error(f"❌ File {i}: {file.name} - Invalid or empty file")
        
        # Execute the workflow
        with st.spinner("🤖 Agents are processing your files..."):
            try:
                # Display file list being processed with detailed info
                file_list = [f.name for f in uploaded_files]
                st.info(f"🔄 Processing {len(uploaded_files)} files: {', '.join(file_list)}")
                st.info(f"📋 File order: {st.session_state.file_order}")
                
                # Show detailed file information
                with st.expander("📁 File Processing Details", expanded=True):
                    st.write(f"**Total uploaded files**: {len(uploaded_files)}")
                    st.write(f"**File order length**: {len(st.session_state.file_order)}")
                    st.write(f"**File order**: {st.session_state.file_order}")
                    
                    for i, file_idx in enumerate(st.session_state.file_order):
                        if file_idx < len(uploaded_files):
                            file = uploaded_files[file_idx]
                            st.write(f"**{i+1}.** {file.name} ({file.type}, {file.size} bytes) - Index: {file_idx}")
                            # Show file ID if available for debugging
                            if hasattr(file, 'file_id'):
                                st.write(f"   File ID: {file.file_id}")
                        else:
                            st.error(f"**{i+1}.** Invalid file index: {file_idx} (max: {len(uploaded_files)-1})")
                
                # Run the complete workflow
                result = st.session_state.workflow_app.invoke(initial_state)
                
                # Debug: Show workflow result structure
                if result.get("error_message"):
                    st.error(f"❌ Workflow returned error: {result['error_message']}")
                    st.session_state.processing_complete = False
                    raise Exception(f"Workflow error: {result['error_message']}")
                
                # Validate result structure before merging
                if not isinstance(result, dict):
                    raise ValueError(f"Workflow returned invalid result type: {type(result)}. Expected dict.")
                
                # Check for required keys and provide detailed info
                required_keys = ["extracted_texts", "processed_chunks", "audio_files"]
                missing_keys = [key for key in required_keys if key not in result]
                if missing_keys:
                    st.warning(f"⚠️ Workflow result missing keys: {missing_keys}")
                    st.json({"workflow_result_keys": list(result.keys())})
                    # Provide default values for missing keys
                    for key in missing_keys:
                        result[key] = []
                
                # Show processing results summary
                extracted_texts = result.get("extracted_texts", [])
                extracted_count = len(extracted_texts)
                chunks_count = len(result.get("processed_chunks", []))
                audio_count = len(result.get("audio_files", []))
                current_step = result.get("current_step", "unknown")
                
                st.info(f"📊 Workflow completed at step '{current_step}': {extracted_count} texts extracted, {chunks_count} chunks processed, {audio_count} audio files generated")
                
                # Critical check: Compare uploaded vs extracted
                uploaded_count = len(uploaded_files)
                if extracted_count < uploaded_count:
                    st.error(f"🚨 PROCESSING INCOMPLETE: {uploaded_count} files uploaded but only {extracted_count} texts extracted!")
                    st.error("This indicates files were skipped during processing. Check console output for details.")
                elif extracted_count == uploaded_count:
                    st.success(f"✅ SUCCESS: All {uploaded_count} files processed correctly!")
                else:
                    st.warning(f"⚠️ UNEXPECTED: {extracted_count} texts extracted from {uploaded_count} uploaded files (includes previous work?)")
                
                # Show detailed extraction results
                if extracted_texts:
                    with st.expander(f"📄 Extraction Results ({extracted_count} files)", expanded=True):
                        for i, text_item in enumerate(extracted_texts):
                            filename = text_item.get("filename", "Unknown")
                            text_length = len(text_item.get("text", ""))
                            file_type = text_item.get("file_type", "Unknown")
                            page = text_item.get("page", i+1)
                            has_error = text_item.get("extraction_error", False)
                            
                            if has_error:
                                st.error(f"**Page {page}**: {filename} ({file_type}) - EXTRACTION FAILED")
                                st.code(text_item.get("text", ""), language="text")
                            else:
                                st.write(f"**Page {page}**: {filename} ({file_type}) - {text_length} characters")
                                if text_length < 10:
                                    st.warning(f"⚠️ Very short text extracted from {filename}")
                                
                                # Show a preview of the text
                                preview = text_item.get("text", "")[:100] + "..." if len(text_item.get("text", "")) > 100 else text_item.get("text", "")
                                st.caption(f"Preview: {preview}")
                else:
                    st.error("❌ No texts were extracted from any files!")
                
                # Merge new results with existing state instead of overwriting
                if "current_state" not in st.session_state:
                    st.session_state.current_state = result
                else:
                    try:
                        # Ensure existing state has all required keys (should already be initialized)
                        required_keys = ["extracted_texts", "processed_chunks", "audio_files"]
                        for key in required_keys:
                            if key not in st.session_state.current_state:
                                st.session_state.current_state[key] = []
                                st.error(f"❌ Critical: Missing key '{key}' in current_state! This indicates a session state corruption.")
                        
                        # Safely append new extracted texts
                        new_extracted = result.get("extracted_texts", [])
                        if new_extracted:
                            st.session_state.current_state["extracted_texts"].extend(new_extracted)
                            st.info(f"✅ Added {len(new_extracted)} new extracted texts")
                        
                        # Safely append new processed chunks
                        new_chunks = result.get("processed_chunks", [])
                        if new_chunks:
                            st.session_state.current_state["processed_chunks"].extend(new_chunks)
                            st.info(f"✅ Added {len(new_chunks)} new text chunks")
                        
                        # Safely append new audio files
                        new_audio = result.get("audio_files", [])
                        if new_audio:
                            st.session_state.current_state["audio_files"].extend(new_audio)
                            st.info(f"✅ Added {len(new_audio)} new audio files")
                        
                        # Update other fields
                        other_fields = {k: v for k, v in result.items() 
                                      if k not in ["extracted_texts", "processed_chunks", "audio_files"]}
                        st.session_state.current_state.update(other_fields)
                        
                        # Show what was updated
                        if other_fields:
                            st.info(f"📝 Updated fields: {', '.join(other_fields.keys())}")
                            
                    except KeyError as ke:
                        st.error(f"❌ KeyError during state merging: Missing key '{ke}' in current_state")
                        st.json({"current_state_keys": list(st.session_state.current_state.keys())})
                        st.json({"workflow_result_keys": list(result.keys())})
                        # Try to recover by reinitializing the state
                        st.warning("🔄 Attempting to recover by reinitializing session state...")
                        st.session_state.current_state = result
                        st.info("✅ Session state reinitialized with current workflow result")
                    except Exception as merge_error:
                        st.error(f"❌ Error during state merging: {str(merge_error)}")
                        st.json({"current_state_type": type(st.session_state.current_state)})
                        st.json({"workflow_result_type": type(result)})
                        # Try to recover by reinitializing the state
                        st.warning("🔄 Attempting to recover by reinitializing session state...")
                        st.session_state.current_state = result
                        st.info("✅ Session state reinitialized with current workflow result")
                
                st.session_state.processing_complete = True
                st.success("✅ All agents completed processing successfully!")
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                
                st.error(f"❌ Agent processing failed: {str(e)}")
                
                # Show detailed error information
                with st.expander("🔍 Detailed Error Information", expanded=True):
                    st.code(error_details, language="python")
                    
                    st.markdown("### 📋 Debug Information")
                    st.markdown(f"**Error Type**: `{type(e).__name__}`")
                    st.markdown(f"**Error Message**: `{str(e)}`")
                    
                    # Show file information
                    if uploaded_files:
                        st.markdown("### 📁 Files Being Processed")
                        for i, file in enumerate(uploaded_files):
                            st.markdown(f"- **File {i+1}**: {file.name} ({file.type}, {file.size} bytes)")
                    
                    # Show current workflow step if available
                    if 'result' in locals() and isinstance(result, dict):
                        st.markdown(f"**Workflow Step**: {result.get('current_step', 'unknown')}")
                        if result.get('error_message'):
                            st.markdown(f"**Workflow Error**: {result['error_message']}")
                    
                    # Show session state info
                    if hasattr(st.session_state, 'current_state') and st.session_state.current_state:
                        existing_files = len(st.session_state.current_state.get("extracted_texts", []))
                        st.markdown(f"**Existing Work**: {existing_files} files previously processed")
                    
                    st.markdown("### 🛠️ Troubleshooting Steps")
                    st.markdown("1. Check if the uploaded files are valid PDF/image files")
                    st.markdown("2. Try uploading one file at a time")
                    st.markdown("3. Use the 'Clear All Work' button to reset the session")
                    st.markdown("4. Check your OpenAI API key if using premium features")
                
                st.session_state.processing_complete = False

# Display results if processing is complete
if st.session_state.processing_complete and st.session_state.current_state:
    current_state = st.session_state.current_state
    
    # Show processing results
    st.header("🤖 Agent Processing Results")
    
    # Display extracted texts
    if current_state.get("extracted_texts"):
        st.subheader("📄 Extracted Texts")
        for item in current_state["extracted_texts"]:
            with st.expander(f"Page {item['page']} - {item['filename']}"):
                st.text(item["text"][:500] + "..." if len(item["text"]) > 500 else item["text"])
    
    # Display processed chunks with enhanced grammar features
    if current_state.get("processed_chunks"):
        st.subheader("📝 Enhanced Text Processing & Grammar Analysis")
        
        # Create tabs for different functionalities
        tab1, tab2, tab3, tab4 = st.tabs(["🔧 Basic Processing", "📝 Parallel Editor", "📊 Grammar Analysis", "🎵 Text-to-Speech"])
        
        with tab1:
            st.markdown("### Basic Text Processing Controls")
            
            # Global actions
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔍 Spell Check All Chunks"):
                    # Check if required credentials are available based on AI provider
                    if actual_spell_check_ai == "OpenAI" and not openai_api_key:
                        st.error("OpenAI API key required for spell check with OpenAI")
                    elif actual_spell_check_ai == "Ollama" and (not ollama_url or not ollama_model):
                        st.error("Ollama URL and model required for spell check with Ollama")
                    else:
                        # Check rate limit before proceeding (OpenAI only)
                        chunks_to_process = len(current_state["processed_chunks"])
                        if actual_spell_check_ai == "OpenAI" and not check_api_rate_limit(chunks_to_process):
                            remaining = get_api_usage_info()["requests_remaining"]
                            st.error(f"❌ Rate limit exceeded! Need {chunks_to_process} requests but only have {remaining} remaining today.")
                        else:
                            with st.spinner(f"Running spell check agent on all chunks using {actual_spell_check_ai}... ({chunks_to_process} requests)" if actual_spell_check_ai == "OpenAI" else f"Running spell check agent on all chunks using {actual_spell_check_ai}..."):
                                text_processor = st.session_state.agents["text_processor"]
                                for i in range(len(current_state["processed_chunks"])):
                                    result_state = text_processor.process_chunk_spell_check(
                                        current_state, i, openai_api_key, actual_spell_check_ai, ollama_url, ollama_model
                                    )
                                    if result_state.get("error_message"):
                                        st.error(f"Spell check failed for chunk {i+1}: {result_state['error_message']}")
                                        break  # Stop on error to avoid wasting more requests
                                    else:
                                        current_state = result_state
                                st.session_state.current_state = current_state
                                st.success("✅ Spell check agent completed for all chunks!")
                                st.rerun()
            
            with col2:
                if st.button("📝 Grammar Check All Chunks"):
                    # Check if required credentials are available based on AI provider
                    if actual_spell_check_ai == "OpenAI" and not openai_api_key:
                        st.error("OpenAI API key required for grammar check with OpenAI")
                    elif actual_spell_check_ai == "Ollama" and (not ollama_url or not ollama_model):
                        st.error("Ollama URL and model required for grammar check with Ollama")
                    else:
                        # Check rate limit before proceeding (OpenAI only)
                        chunks_to_process = len(current_state["processed_chunks"])
                        if actual_spell_check_ai == "OpenAI" and not check_api_rate_limit(chunks_to_process):
                            remaining = get_api_usage_info()["requests_remaining"]
                            st.error(f"❌ Rate limit exceeded! Need {chunks_to_process} requests but only have {remaining} remaining today.")
                        else:
                            with st.spinner(f"Running grammar check agent on all chunks... (Using {chunks_to_process} API requests)"):
                                text_processor = st.session_state.agents["text_processor"]
                                for i in range(len(current_state["processed_chunks"])):
                                    result_state = text_processor.process_chunk_grammar_check(current_state, i)
                                    if result_state.get("error_message"):
                                        st.error(f"Grammar check failed for chunk {i+1}: {result_state['error_message']}")
                                        break  # Stop on error to avoid wasting more requests
                                    else:
                                        current_state = result_state
                                st.session_state.current_state = current_state
                                st.success("✅ Grammar check agent completed for all chunks!")
                                st.rerun()
            
            with col3:
                # Reset all human edits button
                edited_chunks_count = sum(1 for chunk in current_state['processed_chunks'] if chunk.get('human_edited', False))
                if st.button("↩️ Undo All Edits", disabled=edited_chunks_count == 0):
                    with st.spinner("Undoing all human edits..."):
                        text_processor = st.session_state.agents["text_processor"]
                        for i in range(len(current_state["processed_chunks"])):
                            if current_state["processed_chunks"][i].get("human_edited", False):
                                result_state = text_processor.undo_chunk_edit(current_state, i)
                                if not result_state.get("error_message"):
                                    current_state = result_state
                        st.session_state.current_state = current_state
                        st.success("✅ All human edits have been undone!")
                        st.rerun()
        
            # Show editing statistics
            total_chunks = len(current_state['processed_chunks'])
            edited_chunks = sum(1 for chunk in current_state['processed_chunks'] if chunk.get('human_edited', False))
            spell_checked = sum(1 for chunk in current_state['processed_chunks'] if chunk.get('spell_checked', False))
            grammar_checked = sum(1 for chunk in current_state['processed_chunks'] if chunk.get('grammar_checked', False))
            
            st.info(f"📊 **Text Processing Status**: {total_chunks} chunks | Human edited: {edited_chunks} | Spell checked: {spell_checked} | Grammar checked: {grammar_checked}")
            
            # Show editing summary if any chunks have been edited
            if edited_chunks > 0:
                st.success(f"✅ **Human Edits Applied**: {edited_chunks}/{total_chunks} chunks have been manually edited")
        
        with tab2:
            st.markdown("### 📝 Parallel Grammar Editor")
            
            # Chunk selection for parallel editing
            chunks = current_state.get("processed_chunks", [])
            if chunks:
                # Create more informative chunk options showing which files they contain
                chunk_options = []
                for i, chunk in enumerate(chunks):
                    text_preview = chunk['current_text'][:100] + "..." if len(chunk['current_text']) > 100 else chunk['current_text']
                    # Try to identify which file this chunk came from
                    file_markers = []
                    if "--- Page " in chunk['current_text']:
                        import re
                        pages = re.findall(r'--- Page (\d+) ---', chunk['current_text'])
                        if pages:
                            file_markers = [f"Page {p}" for p in pages]
                    
                    if file_markers:
                        chunk_options.append(f"Chunk {i+1}: {', '.join(file_markers)} ({len(chunk['current_text'])} chars)")
                    else:
                        chunk_options.append(f"Chunk {i+1} ({len(chunk['current_text'])} chars)")
                
                # Add option to edit all chunks combined
                chunk_options.append(f"🔗 ALL CHUNKS COMBINED ({sum(len(chunk['current_text']) for chunk in chunks)} total chars)")
                
                selected_option = st.selectbox(
                    "Select text chunk for parallel editing:",
                    range(len(chunk_options)),
                    format_func=lambda x: chunk_options[x],
                    index=len(chunk_options) - 1,  # Default to "ALL CHUNKS COMBINED" (last option)
                    key="parallel_editor_chunk_selection",
                    help="Choose a specific chunk to edit, or select 'ALL CHUNKS COMBINED' to edit all text at once"
                )
                
                # Handle "ALL CHUNKS COMBINED" option
                if selected_option == len(chunks):  # Last option is "ALL CHUNKS COMBINED"
                    selected_chunk_idx = "combined"
                    st.info("🔗 **Combined Mode**: You're editing all extracted text as one document. Changes will be applied to all chunks.")
                else:
                    selected_chunk_idx = selected_option
                
                # Show the parallel editor for the selected chunk
                if selected_chunk_idx is not None:
                    st.markdown("---")
                    
                    # Store TTS settings in session state for parallel editor access
                    if tts_engine == "OpenAI TTS (Premium)":
                        tts_settings = {
                            "voice": openai_voice,
                            "model": openai_model
                        }
                    elif tts_engine == "Mozilla TTS (Local)":
                        tts_settings = {
                            "model": mozilla_model,
                            "vocoder": mozilla_vocoder
                        }
                    elif tts_engine == "Coqui TTS (Local)":
                        tts_settings = {
                            "model": coqui_model,
                            "speaker": coqui_speaker,
                            "language": coqui_language,
                            "reference_audio": coqui_reference_audio.read() if coqui_reference_audio else None
                        }
                    else:
                        tts_settings = {
                            "language": language
                        }
                    
                    st.session_state.current_tts_engine = tts_engine
                    st.session_state.current_tts_settings = tts_settings
                    
                    editor_ui = st.session_state.parallel_editor_ui
                    
                    # Handle combined mode
                    if selected_chunk_idx == "combined":
                        # Create a combined chunk for editing
                        combined_text = "\n\n".join([chunk['current_text'] for chunk in chunks])
                        
                        # Validate combined text structure
                        chunk_lengths = [len(chunk['current_text']) for chunk in chunks]
                        st.info(f"📊 **Text Breakdown**: Chunk lengths: {chunk_lengths} characters each")
                        
                        # Show preview of each chunk's beginning and end
                        with st.expander("🔍 Chunk Content Preview", expanded=False):
                            for i, chunk in enumerate(chunks):
                                text = chunk['current_text']
                                preview_start = text[:100] + "..." if len(text) > 100 else text
                                preview_end = "..." + text[-100:] if len(text) > 100 else ""
                                st.write(f"**Chunk {i+1}** ({len(text)} chars):")
                                st.write(f"Start: {preview_start}")
                                if preview_end:
                                    st.write(f"End: {preview_end}")
                                st.write("---")
                        
                        # Show combined text preview
                        st.info(f"📝 **Combined Preview**: First 200 chars: {combined_text[:200]}...")
                        st.info(f"📝 **Combined Preview**: Last 200 chars: ...{combined_text[-200:]}")
                        
                        # Create a temporary combined state with proper structure
                        combined_chunk = {
                            "id": 0,  # Use numeric ID
                            "original_text": combined_text,
                            "current_text": combined_text,
                            "spell_checked": any(chunk.get("spell_checked", False) for chunk in chunks),
                            "grammar_checked": any(chunk.get("grammar_checked", False) for chunk in chunks),
                            "human_edited": any(chunk.get("human_edited", False) for chunk in chunks),
                            "char_count": len(combined_text),
                            "is_combined": True,  # Mark as combined for reference
                            "original_chunk_count": len(chunks),
                            # Add fields that might be expected by the grammar analyzer
                            "grammar_analysis": {},
                            "completion_suggestions": [],
                            "needs_completion": False
                        }
                        
                        combined_state = {
                            **current_state,
                            "processed_chunks": [combined_chunk]
                        }
                        
                        st.info(f"📄 **Combined View**: Editing {len(chunks)} chunks as one document ({len(combined_text):,} total characters)")
                        st.info("💡 **Tip**: All text from your files is now combined. You can edit everything together and use AI improvements.")
                        
                        # Debug info for combined mode
                        with st.expander("🔧 Debug Info", expanded=False):
                            st.write(f"Combined state has {len(combined_state['processed_chunks'])} chunks")
                            st.write(f"Using API key: {'✅ Provided' if openai_api_key else '❌ Missing'}")
                            st.write(f"Combined text length: {len(combined_text):,} characters")
                        
                        # Use chunk index 0 for the combined chunk
                        try:
                            editor_action = editor_ui.render_full_workflow_interface(
                                combined_state, 0, openai_api_key, ai_provider, ollama_url, ollama_model,
                                actual_summarization_ai
                            )
                        except Exception as e:
                            st.error(f"❌ Error rendering combined parallel editor: {str(e)}")
                            st.error("Please try selecting individual chunks instead of combined mode.")
                            import traceback
                            st.code(traceback.format_exc(), language="python")
                            editor_action = {"action": "error", "error": str(e)}
                    else:
                        editor_action = editor_ui.render_full_workflow_interface(
                            current_state, selected_chunk_idx, openai_api_key, ai_provider, ollama_url, ollama_model,
                            actual_summarization_ai
                        )
                    
                    # Handle parallel editor actions
                    if editor_action and isinstance(editor_action, dict):
                        action_type = editor_action.get("action")
                        
                        if action_type == "approve":
                            # Apply parallel edits to the main state
                            editing_session = editor_action["editing_session"]
                            human_editor = st.session_state.agents.get("human_editor")
                            if human_editor:
                                result_state = human_editor.approve_parallel_edits(current_state, editing_session)
                                if not result_state.get("error_message"):
                                    st.session_state.current_state = result_state
                                    st.success("✅ Parallel edits approved and applied!")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to apply edits: {result_state['error_message']}")
                        
                        elif action_type == "generate_speech" and openai_api_key:
                            # Generate speech from parallel edited text
                            editing_session = editor_action["editing_session"]
                            human_editor = st.session_state.agents.get("human_editor")
                            if human_editor:
                                tts_result = human_editor.generate_speech_from_parallel_edit(
                                    current_state, 
                                    editing_session,
                                    tts_engine,
                                    {"voice": openai_voice if tts_engine == "OpenAI TTS (Premium)" else "alloy"}
                                )
                                
                                if tts_result["success"]:
                                    st.success("🎉 Speech generated from edited text!")
                                    st.audio(tts_result["audio_data"], format="audio/mp3")
                                    
                                    # Use base filename or fallback
                                    base_filename = get_base_filename_from_state()
                                    if base_filename:
                                        audio_filename = f"{base_filename}.wav"
                                        audio_mime = "audio/wav"
                                    else:
                                        audio_filename = f"edited_speech_{selected_chunk_idx}.mp3"
                                        audio_mime = "audio/mp3"
                                    
                                    st.download_button(
                                        "💾 Download Audio",
                                        data=tts_result["audio_data"],
                                        file_name=audio_filename,
                                        mime=audio_mime
                                    )
                                else:
                                    st.error(f"Speech generation failed: {tts_result['error']}")
                        
            else:
                st.warning("No text chunks available for parallel editing")
        
        with tab3:
            st.markdown("### 📊 Advanced Grammar Analysis")
            
            if chunks:
                analysis_chunk_idx = st.selectbox(
                    "Select chunk for detailed grammar analysis:",
                    range(len(chunks)),
                    format_func=lambda x: f"Chunk {x+1} ({len(chunks[x]['current_text'])} chars)",
                    key="grammar_analysis_chunk_selection"
                )
                
                if analysis_chunk_idx is not None:
                    chunk = chunks[analysis_chunk_idx]
                    text = chunk["current_text"]
                    
                    # Perform grammar analysis
                    grammar_agent = st.session_state.grammar_analyzer
                    
                    # Add grammar completeness analysis button
                    if st.button("🔍 Analyze Grammar Completeness", key=f"analyze_grammar_{analysis_chunk_idx}"):
                        with st.spinner("Analyzing sentence completeness..."):
                            analysis = grammar_agent.analyze_sentence_completeness(text)
                            
                            # Store analysis in session state for persistence
                            if 'grammar_analyses' not in st.session_state:
                                st.session_state.grammar_analyses = {}
                            st.session_state.grammar_analyses[analysis_chunk_idx] = analysis
                    
                    # Display analysis if available
                    if hasattr(st.session_state, 'grammar_analyses') and analysis_chunk_idx in st.session_state.grammar_analyses:
                        analysis = st.session_state.grammar_analyses[analysis_chunk_idx]
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Sentences", analysis["total_sentences"])
                        with col2:
                            st.metric("Complete", len(analysis["complete_sentences"]))
                        with col3:
                            st.metric("Incomplete", len(analysis["incomplete_sentences"]))
                        with col4:
                            completeness = analysis["completeness_score"] * 100
                            st.metric("Completeness", f"{completeness:.1f}%")
                        
                        # Show text being analyzed
                        st.subheader("📄 Text Being Analyzed")
                        st.text_area("Current text", text, height=150, disabled=True, key=f"analysis_text_{analysis_chunk_idx}")
                        
                        # Complete sentences
                        if analysis["complete_sentences"]:
                            st.subheader("✅ Complete Sentences")
                            for sentence in analysis["complete_sentences"]:
                                st.success(f"**{sentence['index'] + 1}.** {sentence['text']}")
                        
                        # Incomplete sentences with AI completions
                        if analysis["incomplete_sentences"]:
                            st.subheader("⚠️ Incomplete Sentences")
                            for sentence in analysis["incomplete_sentences"]:
                                issues = ", ".join(sentence["issues"])
                                st.error(f"**{sentence['index'] + 1}.** {sentence['text']}")
                                st.caption(f"Issues: {issues}")
                                
                                # Show LLM completion if API key is provided
                                if openai_api_key:
                                    with st.expander(f"💡 AI Completion Suggestion"):
                                        if st.button(f"Generate Completion", key=f"complete_{analysis_chunk_idx}_{sentence['index']}"):
                                            try:
                                                with st.spinner("Generating AI completion..."):
                                                    completed = grammar_agent.complete_sentence_with_llm(
                                                        sentence["text"], 
                                                        text[:200],  # Context
                                                        openai_api_key
                                                    )
                                                    st.markdown(f"**Original:** {sentence['text']}")
                                                    st.markdown(f"**Completed:** {completed}")
                                                    
                                                    # Option to apply the completion
                                                    if st.button(f"Apply This Completion", key=f"apply_completion_{analysis_chunk_idx}_{sentence['index']}"):
                                                        # Replace in the chunk text
                                                        updated_text = text.replace(sentence["text"], completed)
                                                        
                                                        # Update the chunk
                                                        text_processor = st.session_state.agents["text_processor"]
                                                        result_state = text_processor.process_chunk_human_edit(current_state, analysis_chunk_idx, updated_text)
                                                        if not result_state.get("error_message"):
                                                            st.session_state.current_state = result_state
                                                            st.success("✅ Completion applied to text!")
                                                            st.rerun()
                                                        else:
                                                            st.error(f"Failed to apply completion: {result_state['error_message']}")
                                                        
                                            except Exception as e:
                                                st.error(f"Error generating completion: {str(e)}")
            else:
                st.warning("No text chunks available for grammar analysis")
        
        with tab4:
            st.markdown("### 🎵 Text-to-Speech Generation")
            
            # Generate all audio button
            if st.button("🎵 Generate Speech for All Chunks", type="primary"):
                # Check API key if needed
                if tts_engine == "OpenAI TTS (Premium)" and not openai_api_key:
                    st.error("OpenAI API key required for OpenAI TTS")
                else:
                    # Check rate limit for OpenAI TTS
                    if tts_engine == "OpenAI TTS (Premium)":
                        chunks_to_process = len(current_state["processed_chunks"])
                        if not check_api_rate_limit(chunks_to_process):
                            remaining = get_api_usage_info()["requests_remaining"]
                            st.error(f"❌ Rate limit exceeded! Need {chunks_to_process} requests but only have {remaining} remaining today.")
                        else:
                            with st.spinner(f"TTS agent converting all chunks... (Using {chunks_to_process} API requests)"):
                                tts_agent = st.session_state.agents["tts_agent"]
                                result_state = tts_agent.convert_all_chunks(current_state)
                                if result_state.get("error_message"):
                                    st.error(result_state["error_message"])
                                else:
                                    st.session_state.current_state = result_state
                                    st.success("✅ TTS agent completed for all chunks!")
                                    st.rerun()
                    else:
                        # Google TTS doesn't use API limits
                        with st.spinner("TTS agent converting all chunks..."):
                            tts_agent = st.session_state.agents["tts_agent"]
                            result_state = tts_agent.convert_all_chunks(current_state)
                            if result_state.get("error_message"):
                                st.error(result_state["error_message"])
                            else:
                                st.session_state.current_state = result_state
                                st.success("✅ TTS agent completed for all chunks!")
                                st.rerun()
            
            st.markdown("---")
            
            # Individual chunk TTS controls
            for i, chunk in enumerate(chunks):
                st.subheader(f"🎵 Chunk {i+1} TTS Controls")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Show chunk text preview
                    preview_text = chunk["current_text"][:200] + "..." if len(chunk["current_text"]) > 200 else chunk["current_text"]
                    st.text_area(f"Chunk {i+1} Preview", preview_text, height=100, disabled=True, key=f"tts_preview_{i}")
                
                with col2:
                    # Check if this chunk already has audio
                    existing_audio = None
                    for audio in current_state.get("audio_files", []):
                        if audio["chunk_id"] == i:
                            existing_audio = audio
                            break
                    
                    if existing_audio:
                        st.success(f"✅ Audio available!")
                        st.info(f"Engine: {existing_audio['engine']}")
                        st.audio(existing_audio["audio_data"], format='audio/mp3')
                        
                        # Use base filename or fallback
                        base_filename = get_base_filename_from_state()
                        if base_filename:
                            audio_filename = f"{base_filename}.wav"
                            audio_mime = "audio/wav"
                        else:
                            audio_filename = f"speech_chunk_{i+1}.mp3"
                            audio_mime = "audio/mp3"
                        
                        st.download_button(
                            f"💾 Download",
                            data=existing_audio["audio_data"],
                            file_name=audio_filename,
                            mime=audio_mime,
                            key=f"download_audio_{i}"
                        )
                    else:
                        # Individual chunk TTS button
                        if st.button(f"🎵 Generate Speech", key=f"tts_chunk_tab4_{i}"):
                            # Check API key if needed
                            if tts_engine == "OpenAI TTS (Premium)" and not openai_api_key:
                                st.error("OpenAI API key required for OpenAI TTS")
                            elif tts_engine == "OpenAI TTS (Premium)" and not check_api_rate_limit(1, chunk["current_text"]):
                                usage_info = get_api_usage_info()
                                if usage_info["usage_percentage"] >= 100:
                                    st.error(f"❌ Daily API limit exceeded! {usage_info['requests_remaining']} requests remaining.")
                                else:
                                    st.error(f"❌ Per-minute token limit would be exceeded! Current: {usage_info['tokens_current_minute']}/{usage_info['tokens_per_minute_limit']} tokens")
                            else:
                                api_msg = " (Using 1 API request)" if tts_engine == "OpenAI TTS (Premium)" else ""
                                with st.spinner(f"TTS agent converting chunk {i+1}...{api_msg}"):
                                    tts_agent = st.session_state.agents["tts_agent"]
                                    result_state = tts_agent.convert_chunk(current_state, i)
                                    if result_state.get("error_message"):
                                        st.error(result_state["error_message"])
                                    else:
                                        st.session_state.current_state = result_state
                                        st.success(f"✅ TTS agent completed for chunk {i+1}!")
                                        st.rerun()
                
                st.markdown("---")

else:
    # Show instructions when no files are uploaded
    if not uploaded_files:
        st.info("👆 Upload one or more files to get started with the multi-agent processing system")
    elif not st.session_state.processing_complete:
        st.info("👆 Click 'Process Files with Agents' to start the automated workflow")

# Agent Information Section
with st.expander("🤖 About the Multi-Agent System"):
    st.markdown("""
    This application uses **LangGraph** to orchestrate multiple specialized agents:
    
    **🗂️ File Processing Agent**
    - Validates uploaded files
    - Manages file ordering
    - Prepares files for text extraction
    
    **📄 Text Extraction Agent**
    - Extracts text from PDFs using PyPDF2
    - Performs OCR on images using Tesseract or EasyOCR
    - Combines texts in the specified order
    
    **✏️ Enhanced Text Processing Agent**
    - Splits text into 4000-character chunks
    - Performs spell checking using OpenAI GPT-3.5
    - Performs grammar checking using OpenAI GPT-3.5
    - **Advanced Grammar Analysis**: Analyzes sentence completeness and structure
    - **Parallel Editing Interface**: Side-by-side original/editable text windows
    - **AI-Powered Sentence Completion**: Automatically completes incomplete sentences
    - **Grammar Issue Detection**: Identifies fragments, run-on sentences, missing subjects/predicates
    - **Completeness Scoring**: Provides overall text quality metrics
    - **Human Editing Support**: Allows manual text editing with version control
    - **Undo/Redo**: Maintains edit history for each chunk
    - **Edit Tracking**: Timestamps and tracks all human modifications
    - **Diff Visualization**: Compare original vs edited text with detailed changes
    
    **🎵 Text-to-Speech Agent**
    - Converts text to speech using Google TTS or OpenAI TTS
    - Handles individual chunk conversion
    - Manages batch processing for all chunks
    
    **🔄 Workflow Orchestration**
    - Agents communicate through shared state
    - Error handling and recovery
    - Parallel processing where possible
    
    **📊 Enhanced API Rate Limiting**
    - **Daily Limit**: 20,000 OpenAI API requests per day
    - **Per-Minute Token Limit**: 20,000 tokens per minute (prevents burst usage)
    - **Real-time Monitoring**: Track both daily and per-minute usage in the sidebar
    - **Smart Token Estimation**: Estimates token usage before making requests
    - **Dual Protection**: Prevents exceeding both daily and per-minute limits
    - **Usage Persistence**: Tracks usage across app restarts
    - **Google TTS**: No rate limits (free alternative)
    - **Automatic Throttling**: Prevents API overuse and account suspension
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>📖 Multi-Agent Text Reader App | Powered by LangGraph & Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)