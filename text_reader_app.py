import streamlit as st
import PyPDF2
import pytesseract
from PIL import Image
import pyttsx3
import io
import base64
import tempfile
import os
from gtts import gTTS
import pygame
import openai
from openai import OpenAI
from dotenv import load_dotenv
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Text Reader App",
    page_icon="📖",
    layout="wide"
)

st.title("Text Reader App")
st.markdown("*Upload PDFs, images (JPG, GIF), or paste your own text to listen with natural-sounding voice*")

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def extract_text_from_image_tesseract(image_file):
    """Extract text from image using Tesseract OCR"""
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except pytesseract.TesseractNotFoundError:
        st.error("ERROR: Tesseract OCR is not installed. Please install it from: https://github.com/UB-Mannheim/tesseract/wiki")
        st.info("TIP: After installation, restart the app to use image OCR features.")
        return None
    except Exception as e:
        st.error(f"Error extracting text from image with Tesseract: {str(e)}")
        return None

def extract_text_from_image_easyocr(image_file):
    """Extract text from image using EasyOCR"""
    if not EASYOCR_AVAILABLE:
        st.error("ERROR: EasyOCR is not installed. Please install it with: pip install easyocr")
        return None
        
    try:
        # Initialize EasyOCR reader (cached to avoid reloading)
        if 'easyocr_reader' not in st.session_state:
            st.session_state.easyocr_reader = easyocr.Reader(['en'])
        
        reader = st.session_state.easyocr_reader
        image = Image.open(image_file)
        
        # Convert PIL image to numpy array for EasyOCR
        import numpy as np
        image_array = np.array(image)
        
        # Extract text
        results = reader.readtext(image_array)
        
        # Combine all detected text
        text = "\n".join([result[1] for result in results])
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from image with EasyOCR: {str(e)}")
        return None

def extract_text_from_image(image_file, ocr_engine="Tesseract"):
    """Extract text from image using selected OCR engine"""
    if ocr_engine == "EasyOCR":
        return extract_text_from_image_easyocr(image_file)
    else:
        return extract_text_from_image_tesseract(image_file)

def text_to_speech_gtts(text, language='en'):
    """Convert text to speech using Google TTS"""
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None

def text_to_speech_openai(text, api_key, voice="alloy", model="tts-1-hd"):
    """Convert text to speech using OpenAI TTS"""
    try:
        # OpenAI TTS has a 4096 character limit
        MAX_LENGTH = 4000
        
        if len(text) > MAX_LENGTH:
            st.warning(f"Text is too long ({len(text)} characters). Only the first {MAX_LENGTH} characters will be converted to speech.")
            text = text[:MAX_LENGTH]
        
        client = OpenAI(api_key=api_key)
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text
        )
        
        audio_bytes = io.BytesIO()
        for chunk in response.iter_bytes():
            audio_bytes.write(chunk)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.error(f"Error generating OpenAI speech: {str(e)}")
        return None

def get_audio_download_link(audio_bytes, filename="speech.mp3"):
    """Generate download link for audio file"""
    b64 = base64.b64encode(audio_bytes).decode()
    href = f'<a href="data:audio/mp3;base64,{b64}" download="{filename}">Download Audio File</a>'
    return href

def spell_check_openai(text, api_key):
    """Check and correct spelling using OpenAI GPT"""
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a spelling assistant. Correct only spelling errors while preserving the original meaning, tone, and grammar structure. Return only the corrected text without explanations."},
                {"role": "user", "content": f"Please correct only the spelling errors in this text:\n\n{text}"}
            ],
            max_tokens=4000,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error during spell check: {str(e)}")
        return None

def grammar_check_openai(text, api_key):
    """Check and correct grammar using OpenAI GPT"""
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert grammar and writing assistant. Your task is to:\n1. Fix grammar errors (subject-verb agreement, tense consistency, etc.)\n2. Correct punctuation mistakes\n3. Improve sentence structure and clarity\n4. Fix run-on sentences and fragments\n5. Ensure proper capitalization\n6. Maintain the original meaning and tone\n\nReturn ONLY the corrected text without any explanations, comments, or meta-text."},
                {"role": "user", "content": f"Fix all grammar, punctuation, and sentence structure issues in this text:\n\n{text}"}
            ],
            max_tokens=4000,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error during grammar check: {str(e)}")
        return None

def split_text_into_chunks(text, chunk_size=4000):
    """Split text into chunks of specified size, preferably at sentence boundaries"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    current_pos = 0
    
    while current_pos < len(text):
        # Calculate the end position for this chunk
        end_pos = current_pos + chunk_size
        
        if end_pos >= len(text):
            # Last chunk
            chunks.append(text[current_pos:])
            break
        
        # Try to find a good breaking point (sentence end)
        chunk_text = text[current_pos:end_pos]
        
        # Look for sentence endings in the last 200 characters
        search_start = max(0, len(chunk_text) - 200)
        sentence_endings = ['.', '!', '?', '\n\n']
        
        best_break = -1
        for ending in sentence_endings:
            pos = chunk_text.rfind(ending, search_start)
            if pos > best_break:
                best_break = pos
        
        if best_break > 0:
            # Found a good breaking point
            actual_end = current_pos + best_break + 1
            chunks.append(text[current_pos:actual_end])
            current_pos = actual_end
        else:
            # No good breaking point found, split at word boundary
            words = chunk_text.split()
            if len(words) > 1:
                # Remove the last word to avoid cutting mid-word
                chunk_without_last_word = ' '.join(words[:-1])
                chunks.append(text[current_pos:current_pos + len(chunk_without_last_word)])
                current_pos += len(chunk_without_last_word)
                # Skip any whitespace
                while current_pos < len(text) and text[current_pos].isspace():
                    current_pos += 1
            else:
                # Single very long word, just cut it
                chunks.append(chunk_text)
                current_pos = end_pos
    
    return chunks

# Sidebar for settings
st.sidebar.header("⚙️ Settings")

# TTS Engine Selection
tts_engine = st.sidebar.selectbox(
    "TTS Engine",
    options=["Google TTS (Free)", "OpenAI TTS (Premium)"],
    help="Choose between free Google TTS or premium OpenAI TTS"
)

# OpenAI Settings
if tts_engine == "OpenAI TTS (Premium)":
    # Try to load from environment first
    env_api_key = os.getenv("OPENAI_API_KEY")
    
    if env_api_key:
        # Use environment variable, don't show input field
        openai_api_key = env_api_key
        st.sidebar.success("SUCCESS: Using API key from .env file")
    else:
        # Show input field only if no .env key exists
        openai_api_key = st.sidebar.text_input(
            "OpenAI API Key", 
            type="password",
            help="Enter your OpenAI API key for premium TTS"
        )
    
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

voice_speed = st.sidebar.slider("Reading Speed", 0.5, 2.0, 1.0, 0.1)

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

# Input method selection
st.header("📝 Choose Input Method")
input_method = st.radio(
    "Select how you want to provide text:",
    options=["Upload File", "Paste Text"],
    horizontal=True
)

# Initialize variables outside column scope
extracted_text = ""
edited_text = ""
has_extracted_text = False

# Main content area - single column for file uploads, two columns for paste text
if input_method == "Upload File":
    # Single column for file uploads (chunks will have their own internal columns)
    pass  # No column structure needed
elif input_method == "Paste Text":
    # Two columns for paste text
    col1, col2 = st.columns([1, 1])

if input_method == "Upload File":
    st.header("📁 Upload Files")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'jpg', 'jpeg', 'png', 'gif'],
        help="Upload PDF, JPG, PNG, or GIF files",
        accept_multiple_files=True
    )
    
    extracted_text = ""
    
    if uploaded_files:
        # Show file list and allow reordering
        st.subheader("📋 File Order")
        
        # Initialize file order in session state
        if 'file_order' not in st.session_state or len(st.session_state.file_order) != len(uploaded_files):
            st.session_state.file_order = list(range(len(uploaded_files)))
        
        # Display current order
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
        
        # Extract text button
        if st.button("✅ Extract Text in This Order", type="primary"):
            all_extracted_texts = []
            
            with st.spinner("Extracting text..."):
                for i, file_idx in enumerate(st.session_state.file_order):
                    uploaded_file = uploaded_files[file_idx]
                    file_type = uploaded_file.type
                    
                    if file_type == "application/pdf":
                        text = extract_text_from_pdf(uploaded_file)
                    elif file_type.startswith("image/"):
                        text = extract_text_from_image(uploaded_file, ocr_engine)
                    else:
                        text = None
                    
                    if text:
                        all_extracted_texts.append(f"--- Page {i+1} ---\n{text}\n")
                    else:
                        all_extracted_texts.append(f"--- Page {i+1} ---\n[No text could be extracted]\n")
            
            extracted_text = "\n".join(all_extracted_texts)
            st.session_state.extracted_text = extracted_text
            st.success(f"SUCCESS: Text extracted from {len(uploaded_files)} file(s)!")
        
        # Get extracted text from session state
        extracted_text = st.session_state.get('extracted_text', "")
        has_extracted_text = bool(extracted_text.strip())
        
        if extracted_text:
            
            # Text editing area
            st.subheader("Extracted Text (in upload order)")
            
            # Split text into chunks
            text_chunks = split_text_into_chunks(extracted_text, 4000)
            
            # Global spell and grammar check buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Spell Check All", help="Use OpenAI to check and correct spelling for all chunks"):
                    # Try to get API key from multiple sources
                    api_key = None
                    
                    # First try environment variable
                    api_key = os.getenv("OPENAI_API_KEY")
                    
                    # Then try from OpenAI TTS settings if selected
                    if not api_key and tts_engine == "OpenAI TTS (Premium)" and 'openai_api_key' in locals() and openai_api_key:
                        api_key = openai_api_key
                    
                    # Finally prompt user if no key found
                    if not api_key:
                        api_key = st.text_input("Enter OpenAI API Key for spell check:", type="password", help="Or set OPENAI_API_KEY in .env file")
                    
                    if api_key:
                        with st.spinner("Checking spelling for all chunks..."):
                            corrected_chunks = []
                            for chunk in text_chunks:
                                corrected_chunk = spell_check_openai(chunk, api_key)
                                corrected_chunks.append(corrected_chunk if corrected_chunk else chunk)
                            
                            corrected_text = '\n'.join(corrected_chunks)
                            st.session_state.extracted_text = corrected_text
                            st.success("SUCCESS: Spell check completed for all chunks!")
                            st.rerun()
                    else:
                        st.warning("WARNING: Please enter OpenAI API key or set OPENAI_API_KEY in .env file")

            with col2:
                if st.button("Grammar Check All", help="Use OpenAI to check and correct grammar for all chunks"):
                    # Try to get API key from multiple sources
                    api_key = None
                    
                    # First try environment variable
                    api_key = os.getenv("OPENAI_API_KEY")
                    
                    # Then try from OpenAI TTS settings if selected
                    if not api_key and tts_engine == "OpenAI TTS (Premium)" and 'openai_api_key' in locals() and openai_api_key:
                        api_key = openai_api_key
                    
                    # Finally prompt user if no key found
                    if not api_key:
                        api_key = st.text_input("Enter OpenAI API Key for grammar check:", type="password", help="Or set OPENAI_API_KEY in .env file")
                    
                    if api_key:
                        with st.spinner("Checking grammar for all chunks..."):
                            corrected_chunks = []
                            for chunk in text_chunks:
                                corrected_chunk = grammar_check_openai(chunk, api_key)
                                corrected_chunks.append(corrected_chunk if corrected_chunk else chunk)
                            
                            corrected_text = '\n'.join(corrected_chunks)
                            st.session_state.extracted_text = corrected_text
                            st.success("SUCCESS: Grammar check completed for all chunks!")
                            st.rerun()
                    else:
                        st.warning("WARNING: Please enter OpenAI API key or set OPENAI_API_KEY in .env file")
                
                st.info(f"Text has been split into {len(text_chunks)} chunk(s) of max 4000 characters each")
                
                # Store edited chunks in session state
                if 'edited_chunks' not in st.session_state:
                    st.session_state.edited_chunks = text_chunks.copy()
                
                # Update chunks if text was regenerated
                if len(st.session_state.edited_chunks) != len(text_chunks):
                    st.session_state.edited_chunks = text_chunks.copy()
                
                # Display each chunk with its own TTS controls
                for i, chunk in enumerate(text_chunks):
                    st.subheader(f"Chunk {i+1} of {len(text_chunks)}")
                    
                    # Create two columns for each chunk: text editing and TTS
                    chunk_col1, chunk_col2 = st.columns([1, 1])
                    
                    with chunk_col1:
                        st.write("**Text Editor**")
                        # Individual chunk editing
                        chunk_key = f"chunk_{i}"
                        edited_chunk = st.text_area(
                            f"Edit chunk {i+1}:",
                            value=st.session_state.edited_chunks[i] if i < len(st.session_state.edited_chunks) else chunk,
                            height=200,
                            key=chunk_key,
                            help=f"Edit chunk {i+1} before converting to speech"
                        )
                        
                        # Update session state
                        if i < len(st.session_state.edited_chunks):
                            st.session_state.edited_chunks[i] = edited_chunk
                        else:
                            st.session_state.edited_chunks.append(edited_chunk)
                        
                        # Individual chunk spell and grammar check buttons
                        chunk_check_col1, chunk_check_col2 = st.columns(2)
                        
                        with chunk_check_col1:
                            if st.button(f"Spell Check", key=f"spell_chunk_{i}", help=f"Check spelling for chunk {i+1}"):
                                # Try to get API key from multiple sources
                                api_key = None
                                
                                # First try environment variable
                                api_key = os.getenv("OPENAI_API_KEY")
                                
                                # Then try from OpenAI TTS settings if selected
                                if not api_key and tts_engine == "OpenAI TTS (Premium)" and 'openai_api_key' in locals() and openai_api_key:
                                    api_key = openai_api_key
                                
                                # Finally prompt user if no key found
                                if not api_key:
                                    st.error("Please enter OpenAI API key in sidebar or set OPENAI_API_KEY in .env file")
                                else:
                                    with st.spinner(f"Checking spelling for chunk {i+1}..."):
                                        corrected_chunk = spell_check_openai(edited_chunk, api_key)
                                        if corrected_chunk:
                                            if corrected_chunk.strip() != edited_chunk.strip():
                                                st.session_state.edited_chunks[i] = corrected_chunk
                                                # Also update the main extracted text
                                                st.session_state.extracted_text = '\n'.join(st.session_state.edited_chunks)
                                                st.success(f"SUCCESS: Spell check completed for chunk {i+1}! Text has been updated.")
                                                st.rerun()
                                            else:
                                                st.info(f"No spelling issues found in chunk {i+1}.")
                                        else:
                                            st.error(f"Spell check failed for chunk {i+1}.")
                        
                        with chunk_check_col2:
                            if st.button(f"Grammar Check", key=f"grammar_chunk_{i}", help=f"Check grammar for chunk {i+1}"):
                                # Try to get API key from multiple sources
                                api_key = None
                                
                                # First try environment variable
                                api_key = os.getenv("OPENAI_API_KEY")
                                
                                # Then try from OpenAI TTS settings if selected
                                if not api_key and tts_engine == "OpenAI TTS (Premium)" and 'openai_api_key' in locals() and openai_api_key:
                                    api_key = openai_api_key
                                
                                # Finally prompt user if no key found
                                if not api_key:
                                    st.error("Please enter OpenAI API key in sidebar or set OPENAI_API_KEY in .env file")
                                else:
                                    with st.spinner(f"Checking grammar for chunk {i+1}..."):
                                        corrected_chunk = grammar_check_openai(edited_chunk, api_key)
                                        if corrected_chunk:
                                            if corrected_chunk.strip() != edited_chunk.strip():
                                                st.session_state.edited_chunks[i] = corrected_chunk
                                                # Also update the main extracted text
                                                st.session_state.extracted_text = '\n'.join(st.session_state.edited_chunks)
                                                st.success(f"SUCCESS: Grammar check completed for chunk {i+1}! Text has been updated.")
                                                st.rerun()
                                            else:
                                                st.info(f"No grammar issues found in chunk {i+1}.")
                                        else:
                                            st.error(f"Grammar check failed for chunk {i+1}.")
                        
                        # Character count for this chunk
                        st.caption(f"Chunk {i+1} characters: {len(edited_chunk)}")
                    
                    with chunk_col2:
                        st.write("**Text to Speech**")
                        
                        # Check if OpenAI TTS is selected and validate API key
                        can_generate_chunk = True
                        if tts_engine == "OpenAI TTS (Premium)":
                            if not openai_api_key or not openai_api_key.strip():
                                st.warning("WARNING: Please enter your OpenAI API key in the sidebar to use OpenAI TTS")
                                can_generate_chunk = False
                        
                        if edited_chunk.strip() and can_generate_chunk:
                            # Individual chunk TTS button
                            if st.button(f"Generate Speech", key=f"tts_chunk_{i}"):
                                with st.spinner(f"Generating speech for chunk {i+1}..."):
                                    # Choose TTS engine
                                    if tts_engine == "OpenAI TTS (Premium)":
                                        audio_bytes = text_to_speech_openai(
                                            edited_chunk, 
                                            openai_api_key, 
                                            voice=openai_voice, 
                                            model=openai_model
                                        )
                                    else:
                                        audio_bytes = text_to_speech_gtts(edited_chunk, language)
                                    
                                    if audio_bytes:
                                        st.success(f"SUCCESS: Speech generated for chunk {i+1}!")
                                        
                                        # Show which engine was used
                                        engine_info = "OpenAI TTS" if tts_engine == "OpenAI TTS (Premium)" else "Google TTS"
                                        if tts_engine == "OpenAI TTS (Premium)":
                                            st.info(f"Generated using {engine_info} with {openai_voice} voice ({openai_model})")
                                        else:
                                            st.info(f"Generated using {engine_info} in {language}")
                                        
                                        # Audio player
                                        st.audio(audio_bytes.getvalue(), format='audio/mp3')
                                        
                                        # Download link
                                        st.markdown(
                                            get_audio_download_link(audio_bytes.getvalue(), f"speech_chunk_{i+1}.mp3"),
                                            unsafe_allow_html=True
                                        )
                        else:
                            if not edited_chunk.strip():
                                st.info("No text in this chunk")
                            else:
                                st.info("Enter OpenAI API key to generate speech")
                    
                    st.write("---")  # Separator between chunks
                
                # Total character count
                total_chars = sum(len(chunk) for chunk in st.session_state.edited_chunks)
                st.caption(f"Total characters: {total_chars}")
                
        else:
            st.error("ERROR: No text could be extracted from any file.")

elif input_method == "Paste Text":
    with col1:
        st.header("Paste Your Text")
        edited_text = st.text_area(
            "Paste or type your text here:",
            value="",
            height=300,
            placeholder="Enter the text you want to convert to speech...",
            help="Paste or type any text you want to convert to natural-sounding speech"
        )
        
        if edited_text.strip():
            extracted_text = edited_text
            has_extracted_text = True
            st.success("SUCCESS: Text ready for conversion!")
            # Character count
            st.caption(f"Characters: {len(edited_text)}")
        else:
            st.info("Enter some text to get started")

    with col2:
        st.header("Text to Speech")
        
        # Check if we have text to process for paste text mode
        if has_extracted_text and extracted_text.strip():
            # Check if OpenAI TTS is selected and validate API key
            can_generate = True
            if tts_engine == "OpenAI TTS (Premium)":
                if not openai_api_key or not openai_api_key.strip():
                    st.warning("WARNING: Please enter your OpenAI API key in the sidebar to use OpenAI TTS")
                    can_generate = False
            
            # Single text processing for paste text
            if can_generate and st.button("Generate Speech", type="primary"):
                with st.spinner("Generating speech..."):
                    # Choose TTS engine
                    if tts_engine == "OpenAI TTS (Premium)":
                        audio_bytes = text_to_speech_openai(
                            extracted_text, 
                            openai_api_key, 
                            voice=openai_voice, 
                            model=openai_model
                        )
                    else:
                        audio_bytes = text_to_speech_gtts(extracted_text, language)
                    
                    if audio_bytes:
                        st.success("SUCCESS: Speech generated successfully!")
                        
                        # Show which engine was used
                        engine_info = "OpenAI TTS" if tts_engine == "OpenAI TTS (Premium)" else "Google TTS"
                        if tts_engine == "OpenAI TTS (Premium)":
                            st.info(f"Generated using {engine_info} with {openai_voice} voice ({openai_model})")
                        else:
                            st.info(f"Generated using {engine_info} in {language}")
                        
                        # Audio player
                        st.audio(audio_bytes.getvalue(), format='audio/mp3')
                        
                        # Download link
                        st.markdown(
                            get_audio_download_link(audio_bytes.getvalue()),
                            unsafe_allow_html=True
                        )
                        
                        # Text preview
                        st.subheader("Text Preview")
                        with st.expander("Click to view full text"):
                            st.text(extracted_text)
        else:
            st.info("Enter some text to get started")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>📖 Professional Text Reader App | Supports PDF, JPG, PNG, GIF & Custom Text</p>
    </div>
    """,
    unsafe_allow_html=True
)