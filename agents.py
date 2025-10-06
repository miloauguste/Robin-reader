"""
LangGraph Agents for Text Reader App
Specialized agents for file processing, text extraction, grammar/spell checking, and TTS conversion
"""

import os
import io
import json
import requests
from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime, date
from langgraph.graph import StateGraph, END
import PyPDF2
import pytesseract
from PIL import Image
from openai import OpenAI
from gtts import gTTS
import streamlit as st

# Rate limiting class for OpenAI API
class OpenAIRateLimiter:
    """Rate limiter to enforce daily API request limits and per-minute token limits"""
    
    def __init__(self, daily_limit: int = 20000, tokens_per_minute: int = 20000, storage_file: str = "api_usage.json"):
        self.daily_limit = daily_limit
        self.tokens_per_minute = tokens_per_minute
        self.storage_file = storage_file
        self.usage_data = self._load_usage_data()
        self.minute_tokens = []  # List of (timestamp, token_count) tuples
    
    def _load_usage_data(self) -> Dict[str, Any]:
        """Load usage data from file"""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                return data
            else:
                return {"date": str(date.today()), "requests": 0, "tokens_used": 0}
        except Exception:
            return {"date": str(date.today()), "requests": 0, "tokens_used": 0}
    
    def _save_usage_data(self):
        """Save usage data to file"""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.usage_data, f)
        except Exception:
            pass  # Fail silently if we can't save
    
    def _reset_if_new_day(self):
        """Reset counter if it's a new day"""
        today = str(date.today())
        if self.usage_data.get("date") != today:
            self.usage_data = {"date": today, "requests": 0, "tokens_used": 0}
            self._save_usage_data()
    
    def _clean_old_token_records(self):
        """Remove token usage records older than 1 minute"""
        from datetime import datetime, timedelta
        cutoff_time = datetime.now() - timedelta(minutes=1)
        self.minute_tokens = [
            (timestamp, tokens) for timestamp, tokens in self.minute_tokens
            if timestamp > cutoff_time
        ]
    
    def _get_current_minute_tokens(self) -> int:
        """Get total tokens used in the current minute"""
        self._clean_old_token_records()
        return sum(tokens for _, tokens in self.minute_tokens)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation: 1 token ≈ 4 characters)"""
        return max(1, len(text) // 4)
    
    def can_make_request(self, num_requests: int = 1, text: str = "") -> bool:
        """Check if we can make the requested number of API calls within daily and per-minute limits"""
        self._reset_if_new_day()
        
        # Check daily limit
        if (self.usage_data["requests"] + num_requests) > self.daily_limit:
            return False
        
        # Check per-minute token limit if text is provided
        if text:
            estimated_tokens = self._estimate_tokens(text)
            current_minute_tokens = self._get_current_minute_tokens()
            if (current_minute_tokens + estimated_tokens) > self.tokens_per_minute:
                return False
        
        return True
    
    def record_request(self, num_requests: int = 1, text: str = ""):
        """Record that API requests were made"""
        from datetime import datetime
        
        self._reset_if_new_day()
        self.usage_data["requests"] += num_requests
        
        # Record tokens for per-minute tracking
        if text:
            estimated_tokens = self._estimate_tokens(text)
            self.usage_data["tokens_used"] = self.usage_data.get("tokens_used", 0) + estimated_tokens
            self.minute_tokens.append((datetime.now(), estimated_tokens))
        
        self._save_usage_data()
    
    def get_remaining_requests(self) -> int:
        """Get number of remaining requests for today"""
        self._reset_if_new_day()
        return max(0, self.daily_limit - self.usage_data["requests"])
    
    def get_usage_info(self) -> Dict[str, Any]:
        """Get current usage information"""
        self._reset_if_new_day()
        current_minute_tokens = self._get_current_minute_tokens()
        
        return {
            "date": self.usage_data["date"],
            "requests_used": self.usage_data["requests"],
            "requests_remaining": self.get_remaining_requests(),
            "daily_limit": self.daily_limit,
            "usage_percentage": (self.usage_data["requests"] / self.daily_limit) * 100,
            "tokens_used_today": self.usage_data.get("tokens_used", 0),
            "tokens_current_minute": current_minute_tokens,
            "tokens_per_minute_limit": self.tokens_per_minute,
            "minute_usage_percentage": (current_minute_tokens / self.tokens_per_minute) * 100 if self.tokens_per_minute > 0 else 0
        }

# Global rate limiter instance
rate_limiter = OpenAIRateLimiter(daily_limit=20000)

# Ollama client class for local LLM integration
class OllamaClient:
    """Client for interacting with Ollama local LLM server"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api"
    
    def is_available(self) -> bool:
        """Check if Ollama server is available"""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m['name'] for m in models]
        except Exception:
            pass
        return []
    
    def generate(self, model: str, prompt: str, system_prompt: str = None, timeout: int = 60, **kwargs) -> str:
        """Generate text using Ollama model"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                **kwargs
            }
            
            response = requests.post(f"{self.api_url}/chat", json=payload, timeout=timeout)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('message', {}).get('content', '').strip()
            else:
                raise Exception(f"Ollama request failed: {response.status_code} - {response.text}")
        
        except Exception as e:
            raise Exception(f"Ollama generation error: {str(e)}")

# State definition for the workflow
class TextReaderState(TypedDict):
    uploaded_files: List[Any]  # Streamlit uploaded files
    file_order: List[int]  # Order to process files
    extracted_texts: List[Dict[str, str]]  # List of {page: int, text: str, filename: str}
    processed_chunks: List[Dict[str, Any]]  # Processed text chunks with metadata
    spell_checked: bool
    grammar_checked: bool
    human_edited: bool  # Track if any chunk has been human edited
    audio_files: List[Dict[str, Any]]  # Generated audio files
    current_step: str
    error_message: Optional[str]
    api_key: Optional[str]
    tts_engine: str
    tts_settings: Dict[str, Any]
    ocr_engine: str
    editing_mode: bool  # Whether the workflow is in editing mode

class FileProcessingAgent:
    """Agent responsible for handling uploaded files and determining processing order"""
    
    def __init__(self):
        self.name = "file_processor"
    
    def __call__(self, state: TextReaderState) -> TextReaderState:
        """Process uploaded files and set up file order"""
        try:
            uploaded_files = state.get("uploaded_files", [])
            
            if not uploaded_files:
                return {
                    **state,
                    "error_message": "No files uploaded",
                    "current_step": "error"
                }
            
            # Initialize file order if not set
            file_order = state.get("file_order")
            if not file_order or len(file_order) != len(uploaded_files):
                file_order = list(range(len(uploaded_files)))
            
            # Validate file types
            supported_types = ['application/pdf', 'image/jpeg', 'image/jpg', 'image/png', 'image/gif']
            valid_files = []
            
            for i, file in enumerate(uploaded_files):
                if file.type in supported_types:
                    valid_files.append({
                        'index': i,
                        'name': file.name,
                        'type': file.type,
                        'file_object': file
                    })
            
            if not valid_files:
                return {
                    **state,
                    "error_message": "No supported file types found. Please upload PDF, JPG, PNG, or GIF files.",
                    "current_step": "error"
                }
            
            return {
                **state,
                "file_order": file_order,
                "uploaded_files": uploaded_files,
                "current_step": "text_extraction",
                "error_message": None
            }
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            
            # Provide detailed error information about which files failed
            uploaded_files = state.get("uploaded_files", [])
            file_info = []
            for i, file in enumerate(uploaded_files):
                if hasattr(file, 'name') and hasattr(file, 'type') and hasattr(file, 'size'):
                    file_info.append(f"File {i+1}: {file.name} ({file.type}, {file.size} bytes)")
                else:
                    file_info.append(f"File {i+1}: {str(file)}")
            
            detailed_error = f"File processing validation failed for {len(uploaded_files)} uploaded files. Error: {str(e)}"
            
            return {
                **state,
                "error_message": detailed_error,
                "error_details": error_details,
                "file_info": file_info,
                "current_step": "error"
            }

class TextExtractionAgent:
    """Agent responsible for extracting text from files using OCR and PDF parsing"""
    
    def __init__(self):
        self.name = "text_extractor"
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
    def extract_text_from_image_tesseract(self, image_file) -> str:
        """Extract text from image using Tesseract OCR"""
        try:
            image = Image.open(image_file)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except pytesseract.TesseractNotFoundError:
            return "Error: Tesseract OCR not installed"
        except Exception as e:
            return f"Error extracting text from image: {str(e)}"
    
    def extract_text_from_image_easyocr(self, image_file) -> str:
        """Extract text from image using EasyOCR"""
        try:
            import easyocr
            import numpy as np
            
            # Initialize reader (should be cached in production)
            reader = easyocr.Reader(['en'])
            image = Image.open(image_file)
            image_array = np.array(image)
            results = reader.readtext(image_array)
            text = "\n".join([result[1] for result in results])
            return text.strip()
        except ImportError:
            return "Error: EasyOCR not installed"
        except Exception as e:
            return f"Error extracting text with EasyOCR: {str(e)}"
    
    def __call__(self, state: TextReaderState) -> TextReaderState:
        """Extract text from all uploaded files in specified order"""
        import streamlit as st
        import time
        
        try:
            uploaded_files = state.get("uploaded_files", [])
            file_order = state.get("file_order", [])
            ocr_engine = state.get("ocr_engine", "Tesseract")
            
            # Get existing extracted texts to append to
            existing_extracted_texts = state.get("extracted_texts", [])
            extracted_texts = existing_extracted_texts.copy()  # Start with existing work
            
            # Calculate starting page number based on existing work
            max_existing_page = max([item.get("page", 0) for item in existing_extracted_texts], default=0)
            
            # Check if we're processing new files or duplicates
            existing_filenames = {item["filename"] for item in existing_extracted_texts}
            new_extractions = []
            
            # Calculate files to process (skip already processed)
            files_to_process = [
                (position_index, file_idx) for position_index, file_idx in enumerate(file_order)
                if file_idx < len(uploaded_files) and uploaded_files[file_idx].name not in existing_filenames
            ]
            
            print(f"DEBUG: Starting file extraction")
            print(f"DEBUG: uploaded_files count: {len(uploaded_files)}")
            print(f"DEBUG: files to process: {len(files_to_process)}")
            print(f"DEBUG: existing_filenames: {existing_filenames}")
            
            # Create progress bar and time estimation
            if files_to_process:
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_text = st.empty()
                start_time = time.time()
                
                # Time estimates per file type (seconds)
                time_estimates = {
                    "application/pdf": 5,  # 5 seconds per PDF page
                    "image/jpeg": 15,      # 15 seconds per image (OCR)
                    "image/png": 15,
                    "image/gif": 15,
                    "default": 10
                }
            
            for i, (position_index, file_idx) in enumerate(files_to_process):
                # Update progress
                progress = (i + 1) / len(files_to_process)
                progress_bar.progress(progress)
                
                uploaded_file = uploaded_files[file_idx]
                file_type = uploaded_file.type
                
                # Estimate time for this file
                estimated_time = time_estimates.get(file_type, time_estimates["default"])
                
                # Update status
                status_text.text(f"📄 Processing file {i + 1} of {len(files_to_process)}: {uploaded_file.name}")
                
                # Calculate time estimates
                elapsed_time = time.time() - start_time
                if i > 0:
                    avg_time_per_file = elapsed_time / i
                    remaining_files = len(files_to_process) - i
                    estimated_remaining = remaining_files * avg_time_per_file
                    time_text.text(f"⏱️ Elapsed: {elapsed_time:.1f}s | Estimated remaining: {estimated_remaining:.1f}s")
                else:
                    time_text.text(f"⏱️ Starting extraction... Estimated: {estimated_time}s per file")
                
                print(f"Processing file {i + 1}/{len(files_to_process)}: {uploaded_file.name} (type: {file_type})")
                
                # Extract text based on file type
                file_start_time = time.time()
                
                try:
                    if file_type == "application/pdf":
                        text = self.extract_text_from_pdf(uploaded_file)
                    elif file_type.startswith("image/"):
                        if ocr_engine == "EasyOCR":
                            text = self.extract_text_from_image_easyocr(uploaded_file)
                        else:
                            text = self.extract_text_from_image_tesseract(uploaded_file)
                    else:
                        text = "[Unsupported file type]"
                    
                    print(f"Extracted {len(text)} characters from {uploaded_file.name}")
                    
                    new_extraction = {
                        "page": max_existing_page + len(new_extractions) + 1,
                        "text": text,
                        "filename": uploaded_file.name,
                        "file_type": file_type
                    }
                    
                    new_extractions.append(new_extraction)
                    print(f"DEBUG: Added extraction for {uploaded_file.name}, total new extractions: {len(new_extractions)}")
                    
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    print(f"ERROR: Failed to extract from {uploaded_file.name}: {str(e)}")
                    print(f"ERROR: Full traceback:\n{error_details}")
                    # Add failed extraction to list to track it
                    new_extractions.append({
                        "page": max_existing_page + len(new_extractions) + 1,
                        "text": f"[ERROR: Failed to extract text from {uploaded_file.name}: {str(e)}]",
                        "filename": uploaded_file.name,
                        "file_type": file_type,
                        "extraction_error": True
                    })
                    print(f"DEBUG: Added error placeholder for {uploaded_file.name}, total new extractions: {len(new_extractions)}")
                    # Continue with next file instead of stopping
            
            # Complete progress tracking
            if files_to_process:
                progress_bar.progress(1.0)
                total_time = time.time() - start_time
                status_text.text(f"✅ Completed processing {len(files_to_process)} files!")
                time_text.text(f"⏱️ Total time: {total_time:.1f}s | Average: {total_time/len(files_to_process):.1f}s per file")
            
            # Add new extractions to the list
            extracted_texts.extend(new_extractions)
            
            print(f"DEBUG: Final extraction summary:")
            print(f"DEBUG: - new_extractions count: {len(new_extractions)}")
            print(f"DEBUG: - total extracted_texts count: {len(extracted_texts)}")
            print(f"DEBUG: - filenames processed: {[ext['filename'] for ext in new_extractions]}")
            
            # If no new files were processed and no existing files, return error
            if not extracted_texts:
                return {
                    **state,
                    "error_message": "No text could be extracted from any file",
                    "current_step": "error"
                }
            
            # If no new files were processed but existing files exist, skip to next step
            if not new_extractions and existing_extracted_texts:
                return {
                    **state,
                    "extracted_texts": extracted_texts,
                    "current_step": "text_processing",
                    "error_message": None
                }
            
            return {
                **state,
                "extracted_texts": extracted_texts,
                "current_step": "text_processing",
                "error_message": None
            }
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            
            # Provide detailed error information
            uploaded_files = state.get("uploaded_files", [])
            file_names = [f.name if hasattr(f, 'name') else str(f) for f in uploaded_files]
            
            detailed_error = f"Text extraction failed for files: {', '.join(file_names)}. Error: {str(e)}"
            
            return {
                **state,
                "error_message": detailed_error,
                "error_details": error_details,
                "failed_files": file_names,
                "current_step": "error"
            }

class GrammarAnalysisAgent:
    """Agent responsible for analyzing sentence completeness and providing LLM-based completion"""
    
    def __init__(self):
        self.name = "grammar_analyzer"
        self.sentence_patterns = [
            r'[.!?]\s*$',  # Complete sentences ending with punctuation
            r'[.!?]\s+[A-Z]',  # Sentence boundaries
        ]
    
    def analyze_sentence_completeness(self, text: str) -> Dict[str, Any]:
        """Analyze text for sentence completeness and structural issues"""
        import re
        
        sentences = re.split(r'[.!?]+', text.strip())
        incomplete_sentences = []
        complete_sentences = []
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Basic completeness checks
            has_subject = self._has_subject(sentence)
            has_predicate = self._has_predicate(sentence)
            is_fragment = self._is_fragment(sentence)
            is_run_on = self._is_run_on_sentence(sentence)
            
            sentence_analysis = {
                'index': i,
                'text': sentence,
                'has_subject': has_subject,
                'has_predicate': has_predicate,
                'is_complete': has_subject and has_predicate and not is_fragment,
                'is_fragment': is_fragment,
                'is_run_on': is_run_on,
                'issues': []
            }
            
            # Identify issues
            if not has_subject:
                sentence_analysis['issues'].append('missing_subject')
            if not has_predicate:
                sentence_analysis['issues'].append('missing_predicate')
            if is_fragment:
                sentence_analysis['issues'].append('sentence_fragment')
            if is_run_on:
                sentence_analysis['issues'].append('run_on_sentence')
            
            if sentence_analysis['is_complete'] and not sentence_analysis['issues']:
                complete_sentences.append(sentence_analysis)
            else:
                incomplete_sentences.append(sentence_analysis)
        
        return {
            'total_sentences': len(sentences) - 1,  # Subtract empty last split
            'complete_sentences': complete_sentences,
            'incomplete_sentences': incomplete_sentences,
            'completeness_score': len(complete_sentences) / max(1, len(complete_sentences) + len(incomplete_sentences))
        }
    
    def _has_subject(self, sentence: str) -> bool:
        """Check if sentence has a subject (basic noun/pronoun detection)"""
        import re
        # Simple subject detection - look for common pronouns or capitalized words
        subject_patterns = [
            r'\b(I|you|he|she|it|we|they|this|that|there)\b',
            r'\b[A-Z][a-z]+\b',  # Proper nouns
            r'\bthe\s+\w+\b',    # Definite articles with nouns
            r'\ba\s+\w+\b',      # Indefinite articles with nouns
        ]
        return any(re.search(pattern, sentence, re.IGNORECASE) for pattern in subject_patterns)
    
    def _has_predicate(self, sentence: str) -> bool:
        """Check if sentence has a predicate (basic verb detection)"""
        import re
        # Simple verb detection
        verb_patterns = [
            r'\b(is|are|was|were|am|be|been|being)\b',  # To be verbs
            r'\b(have|has|had)\b',  # Have verbs
            r'\b(do|does|did)\b',   # Do verbs
            r'\b(will|would|can|could|should|shall|may|might|must)\b',  # Modals
            r'\b\w+ed\b',  # Past tense
            r'\b\w+ing\b', # Present participle
            r'\b\w+s\b',   # Third person singular
        ]
        return any(re.search(pattern, sentence, re.IGNORECASE) for pattern in verb_patterns)
    
    def _is_fragment(self, sentence: str) -> bool:
        """Check if text is a sentence fragment"""
        # Very basic fragment detection
        words = sentence.split()
        if len(words) < 2:
            return True
        
        # Check for subordinating conjunctions at the beginning (common fragments)
        subordinating_conjunctions = [
            'although', 'because', 'before', 'if', 'since', 'unless', 
            'until', 'when', 'whenever', 'where', 'while'
        ]
        first_word = words[0].lower()
        return first_word in subordinating_conjunctions
    
    def _is_run_on_sentence(self, sentence: str) -> bool:
        """Check if sentence is a run-on (very basic detection)"""
        # Simple run-on detection based on length and comma count
        words = sentence.split()
        comma_count = sentence.count(',')
        
        # Very long sentences with multiple commas might be run-ons
        return len(words) > 30 and comma_count > 3
    
    def complete_sentence_with_llm(self, incomplete_sentence: str, context: str, api_key: str = None, 
                                   ai_provider: str = "OpenAI", ollama_url: str = None, ollama_model: str = None) -> str:
        """Use LLM to complete or fix incomplete sentences"""
        try:
            # Combine text for rate limit check (only for OpenAI)
            combined_text = f"{context} {incomplete_sentence}"
            
            if ai_provider == "OpenAI":
                if not api_key:
                    raise Exception("OpenAI API key is required for OpenAI provider")
                
                if not rate_limiter.can_make_request(1, combined_text):
                    usage_info = rate_limiter.get_usage_info()
                    if usage_info["usage_percentage"] >= 100:
                        raise Exception(f"Daily API limit exceeded. Remaining requests: {usage_info['requests_remaining']}/{rate_limiter.daily_limit}")
                    else:
                        raise Exception(f"Per-minute token limit would be exceeded. Current minute usage: {usage_info['tokens_current_minute']}/{rate_limiter.tokens_per_minute} tokens")
            
            # Create a detailed prompt for sentence completion
            system_prompt = """You are a grammar expert specializing in sentence completion and repair. Your task is to:
1. Analyze incomplete sentences and sentence fragments
2. Complete or repair them while preserving the original meaning and intent
3. Maintain the original writing style and tone
4. Fill in missing subjects, predicates, or other essential sentence elements
5. Fix run-on sentences by breaking them into proper sentences
6. Ensure grammatical correctness and clarity

Return ONLY the corrected/completed sentence(s) without explanations."""
            
            user_prompt = f"""Context: {context}

Incomplete sentence to fix: "{incomplete_sentence}"

Please complete or repair this sentence to make it grammatically correct and complete while preserving its original meaning."""
            
            # Use appropriate AI provider
            if ai_provider == "OpenAI":
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=500,
                    temperature=0.3
                )
                rate_limiter.record_request(1, combined_text)
                return response.choices[0].message.content.strip()
                
            elif ai_provider == "Ollama":
                if not ollama_url or not ollama_model:
                    raise Exception("Ollama URL and model are required for Ollama provider")
                
                ollama_client = OllamaClient(ollama_url)
                if not ollama_client.is_available():
                    raise Exception(f"Ollama server not available at {ollama_url}")
                
                result = ollama_client.generate(
                    model=ollama_model,
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    timeout=timeout,  # Use configurable timeout
                    temperature=0.3,
                    max_tokens=500
                )
                return result
            
            else:
                raise Exception(f"Unsupported AI provider: {ai_provider}")
            
        except Exception as e:
            raise Exception(f"LLM sentence completion error: {str(e)}")
    
    def process_text_completeness(self, state: TextReaderState, chunk_id: int) -> TextReaderState:
        """Analyze and optionally complete sentences in a text chunk"""
        try:
            processed_chunks = state.get("processed_chunks", [])
            api_key = state.get("api_key")
            
            if chunk_id >= len(processed_chunks):
                raise Exception("Invalid chunk ID")
            
            chunk = processed_chunks[chunk_id]
            text = chunk["current_text"]
            
            # Analyze sentence completeness
            analysis = self.analyze_sentence_completeness(text)
            
            # Store analysis in chunk
            processed_chunks[chunk_id]["grammar_analysis"] = analysis
            processed_chunks[chunk_id]["needs_completion"] = len(analysis['incomplete_sentences']) > 0
            
            # If API key is provided and there are incomplete sentences, offer completions
            if api_key and analysis['incomplete_sentences']:
                completed_suggestions = []
                for incomplete in analysis['incomplete_sentences']:
                    try:
                        completed = self.complete_sentence_with_llm(
                            incomplete['text'], 
                            text[:200],  # Provide context
                            api_key
                        )
                        completed_suggestions.append({
                            'original': incomplete['text'],
                            'completed': completed,
                            'issues': incomplete['issues']
                        })
                    except Exception as e:
                        completed_suggestions.append({
                            'original': incomplete['text'],
                            'completed': incomplete['text'],  # Fallback to original
                            'issues': incomplete['issues'],
                            'completion_error': str(e)
                        })
                
                processed_chunks[chunk_id]["completion_suggestions"] = completed_suggestions
            
            return {
                **state,
                "processed_chunks": processed_chunks,
                "error_message": None
            }
            
        except Exception as e:
            return {
                **state,
                "error_message": f"Grammar analysis error: {str(e)}"
            }

class TextProcessingAgent:
    """Agent responsible for grammar checking, spell checking, and text chunking"""
    
    def __init__(self):
        self.name = "text_processor"
        self.chunk_size = 4000
        self.grammar_analyzer = GrammarAnalysisAgent()
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks of specified size, preferably at sentence boundaries"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            end_pos = current_pos + self.chunk_size
            
            if end_pos >= len(text):
                chunks.append(text[current_pos:])
                break
            
            chunk_text = text[current_pos:end_pos]
            
            # Find good breaking point
            search_start = max(0, len(chunk_text) - 200)
            sentence_endings = ['.', '!', '?', '\n\n']
            
            best_break = -1
            for ending in sentence_endings:
                pos = chunk_text.rfind(ending, search_start)
                if pos > best_break:
                    best_break = pos
            
            if best_break > 0:
                actual_end = current_pos + best_break + 1
                chunks.append(text[current_pos:actual_end])
                current_pos = actual_end
            else:
                words = chunk_text.split()
                if len(words) > 1:
                    chunk_without_last_word = ' '.join(words[:-1])
                    chunks.append(text[current_pos:current_pos + len(chunk_without_last_word)])
                    current_pos += len(chunk_without_last_word)
                    while current_pos < len(text) and text[current_pos].isspace():
                        current_pos += 1
                else:
                    chunks.append(chunk_text)
                    current_pos = end_pos
        
        return chunks
    
    def spell_check_with_ai(self, text: str, api_key: str = None, ai_provider: str = "OpenAI", 
                          ollama_url: str = None, ollama_model: str = None, timeout: int = 90) -> str:
        """Check and correct spelling using AI (OpenAI or Ollama) with rate limiting"""
        try:
            # Check rate limit for OpenAI only
            if ai_provider == "OpenAI":
                if not api_key:
                    raise Exception("OpenAI API key is required for OpenAI provider")
                    
                if not rate_limiter.can_make_request(1, text):
                    usage_info = rate_limiter.get_usage_info()
                    if usage_info["usage_percentage"] >= 100:
                        raise Exception(f"Daily API limit exceeded. Remaining requests: {usage_info['requests_remaining']}/{rate_limiter.daily_limit}")
                    else:
                        raise Exception(f"Per-minute token limit would be exceeded. Current minute usage: {usage_info['tokens_current_minute']}/{rate_limiter.tokens_per_minute} tokens")
            
            system_prompt = "You are a spelling assistant. Correct only spelling errors while preserving the original meaning, tone, and grammar structure. Return only the corrected text without explanations."
            user_prompt = f"Please correct only the spelling errors in this text:\n\n{text}"
            
            # Use appropriate AI provider
            if ai_provider == "OpenAI":
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=4000,
                    temperature=0.1
                )
                rate_limiter.record_request(1, text)
                return response.choices[0].message.content.strip()
                
            elif ai_provider == "Ollama":
                if not ollama_url or not ollama_model:
                    raise Exception("Ollama URL and model are required for Ollama provider")
                
                ollama_client = OllamaClient(ollama_url)
                if not ollama_client.is_available():
                    raise Exception(f"Ollama server not available at {ollama_url}")
                
                result = ollama_client.generate(
                    model=ollama_model,
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    timeout=90,  # 1.5 minutes timeout for spell checking  
                    temperature=0.1
                )
                return result
            
            else:
                raise Exception(f"Unsupported AI provider: {ai_provider}")
        except Exception as e:
            raise Exception(f"Spell check error: {str(e)}")
    
    # Backward compatibility method
    def spell_check_openai(self, text: str, api_key: str) -> str:
        """Backward compatibility wrapper for OpenAI spell check"""
        return self.spell_check_with_ai(text, api_key, "OpenAI")
    
    def grammar_check_openai(self, text: str, api_key: str) -> str:
        """Check and correct grammar using OpenAI GPT with rate limiting"""
        try:
            # Check rate limit before making request (both daily and per-minute)
            if not rate_limiter.can_make_request(1, text):
                usage_info = rate_limiter.get_usage_info()
                if usage_info["usage_percentage"] >= 100:
                    raise Exception(f"Daily API limit exceeded. Remaining requests: {usage_info['requests_remaining']}/{rate_limiter.daily_limit}")
                else:
                    raise Exception(f"Per-minute token limit would be exceeded. Current minute usage: {usage_info['tokens_current_minute']}/{rate_limiter.tokens_per_minute} tokens")
            
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
            
            # Record the successful request with text for token tracking
            rate_limiter.record_request(1, text)
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"Grammar check error: {str(e)}")
    
    def improve_text_with_ai(self, text: str, api_key: str = None, improvement_type: str = "general", 
                           ai_provider: str = "OpenAI", ollama_url: str = None, ollama_model: str = None, timeout: int = 120) -> str:
        """Improve text using AI (OpenAI or Ollama) with specific improvement types"""
        try:
            # Check rate limit for OpenAI only
            if ai_provider == "OpenAI":
                if not api_key:
                    raise Exception("OpenAI API key is required for OpenAI provider")
                    
                if not rate_limiter.can_make_request(1, text):
                    usage_info = rate_limiter.get_usage_info()
                    if usage_info["usage_percentage"] >= 100:
                        raise Exception(f"Daily API limit exceeded. Remaining requests: {usage_info['requests_remaining']}/{rate_limiter.daily_limit}")
                    else:
                        raise Exception(f"Per-minute token limit would be exceeded. Current minute usage: {usage_info['tokens_current_minute']}/{rate_limiter.tokens_per_minute} tokens")
            
            # Define improvement prompts
            improvement_prompts = {
                "general": "Improve the text for clarity, readability, and flow while preserving the original meaning and tone. Fix grammar, spelling, and sentence structure issues.",
                "professional": "Rewrite the text in a more professional and polished tone while maintaining the original meaning. Improve clarity and formality.",
                "casual": "Rewrite the text in a more casual and conversational tone while maintaining the original meaning. Make it more approachable and friendly.",
                "concise": "Make the text more concise and direct while preserving all important information. Remove redundancy and improve clarity.",
                "detailed": "Expand the text with more detail and explanation while maintaining the original meaning. Add context and clarification where helpful.",
                "formal": "Rewrite the text in a formal academic or business tone while preserving the original meaning and information."
            }
            
            system_prompt = improvement_prompts.get(improvement_type, improvement_prompts["general"])
            
            # Use appropriate AI provider
            if ai_provider == "OpenAI":
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt + " Return only the improved text without explanations or additional commentary."},
                        {"role": "user", "content": text}
                    ],
                    max_tokens=4000,
                    temperature=0.4
                )
                rate_limiter.record_request(1, text)
                return response.choices[0].message.content.strip()
                
            elif ai_provider == "Ollama":
                if not ollama_url or not ollama_model:
                    raise Exception("Ollama URL and model are required for Ollama provider")
                
                ollama_client = OllamaClient(ollama_url)
                if not ollama_client.is_available():
                    raise Exception(f"Ollama server not available at {ollama_url}")
                
                result = ollama_client.generate(
                    model=ollama_model,
                    prompt=text,
                    system_prompt=system_prompt + " Return only the improved text without explanations or additional commentary.",
                    timeout=timeout,  # Use configurable timeout
                    temperature=0.4
                )
                return result
            
            else:
                raise Exception(f"Unsupported AI provider: {ai_provider}")
        except Exception as e:
            raise Exception(f"Text improvement error: {str(e)}")
    
    def summarize_text_with_ai(self, text: str, api_key: str = None, target_length: int = 350,
                             ai_provider: str = "OpenAI", ollama_url: str = None, ollama_model: str = None, timeout: int = 180) -> str:
        """Summarize text using AI (OpenAI or Ollama) with specific line references and target word count"""
        try:
            # Check rate limit for OpenAI only
            if ai_provider == "OpenAI":
                if not api_key:
                    raise Exception("OpenAI API key is required for OpenAI provider")
                    
                if not rate_limiter.can_make_request(1, text):
                    usage_info = rate_limiter.get_usage_info()
                    if usage_info["usage_percentage"] >= 100:
                        raise Exception(f"Daily API limit exceeded. Remaining requests: {usage_info['requests_remaining']}/{rate_limiter.daily_limit}")
                    else:
                        raise Exception(f"Per-minute token limit would be exceeded. Current minute usage: {usage_info['tokens_current_minute']}/{rate_limiter.tokens_per_minute} tokens")
            
            # Add line numbers to the text for reference
            lines = text.split('\n')
            numbered_text = ""
            for i, line in enumerate(lines, 1):
                if line.strip():  # Only add line numbers to non-empty lines
                    numbered_text += f"Line {i}: {line}\n"
                else:
                    numbered_text += "\n"
            
            system_prompt = f"""You are an expert text summarizer. Create a comprehensive summary of the provided text that:

1. Is approximately {target_length} words or more in length
2. Captures all key points, themes, and important details from the text
3. References specific line numbers when discussing content (e.g., "As mentioned in Line 5..." or "Line 12 discusses...")
4. Maintains logical flow and coherence
5. Provides context and analysis, not just a list of points
6. Uses clear, engaging prose suitable for someone who hasn't read the original text

CRITICAL FORMATTING REQUIREMENT: Your entire response must be written in well-structured paragraph form only. Do NOT use:
- Bullet points or numbered lists
- Headers or subheadings
- Dashes or other list-style formatting
- Line breaks that create artificial separation between related ideas

Write your summary as continuous, flowing paragraphs with smooth transitions between ideas. Each paragraph should focus on related themes and naturally lead to the next. Use embedded line references within the paragraph text."""
            
            user_prompt = f"Please create a detailed summary of the following text with line references:\n\n{numbered_text}"
            
            # Use appropriate AI provider
            if ai_provider == "OpenAI":
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=4000,
                    temperature=0.3
                )
                rate_limiter.record_request(1, text)
                return response.choices[0].message.content.strip()
                
            elif ai_provider == "Ollama":
                if not ollama_url or not ollama_model:
                    raise Exception("Ollama URL and model are required for Ollama provider")
                
                ollama_client = OllamaClient(ollama_url)
                if not ollama_client.is_available():
                    raise Exception(f"Ollama server not available at {ollama_url}")
                
                result = ollama_client.generate(
                    model=ollama_model,
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    timeout=timeout,  # Use configurable timeout
                    temperature=0.3
                )
                return result
            
            else:
                raise Exception(f"Unsupported AI provider: {ai_provider}")
        except Exception as e:
            raise Exception(f"Text summarization error: {str(e)}")
    
    # Backward compatibility methods
    def improve_text_openai(self, text: str, api_key: str, improvement_type: str = "general") -> str:
        """Backward compatibility wrapper for OpenAI text improvement"""
        return self.improve_text_with_ai(text, api_key, improvement_type, "OpenAI")
    
    def summarize_text_openai(self, text: str, api_key: str, target_length: int = 350) -> str:
        """Backward compatibility wrapper for OpenAI text summarization"""
        return self.summarize_text_with_ai(text, api_key, target_length, "OpenAI")
    
    def __call__(self, state: TextReaderState) -> TextReaderState:
        """Process extracted texts into chunks and optionally perform spell/grammar checking"""
        try:
            extracted_texts = state.get("extracted_texts", [])
            existing_processed_chunks = state.get("processed_chunks", [])
            api_key = state.get("api_key")
            
            # Get existing processed filenames to avoid reprocessing
            existing_chunk_sources = set()
            for chunk in existing_processed_chunks:
                # Extract page info from chunk text if it exists
                chunk_text = chunk.get("original_text", "")
                if chunk_text.startswith("--- Page "):
                    existing_chunk_sources.add(chunk_text.split("\n")[0])
            
            # Only process new extracted texts that haven't been chunked yet
            new_extracted_texts = []
            for item in extracted_texts:
                page_marker = f"--- Page {item['page']} ---"
                if page_marker not in existing_chunk_sources:
                    new_extracted_texts.append(item)
            
            # If no new texts to process, return existing state
            if not new_extracted_texts and existing_processed_chunks:
                return {
                    **state,
                    "processed_chunks": existing_processed_chunks,
                    "current_step": "ready_for_tts",
                    "error_message": None
                }
            
            # Combine only new extracted texts
            new_combined_text = "\n".join([
                f"--- Page {item['page']} ---\n{item['text']}\n"
                for item in new_extracted_texts
            ])
            
            # Split new text into chunks
            new_chunks = self.split_text_into_chunks(new_combined_text) if new_combined_text.strip() else []
            
            # Start chunk IDs from where existing chunks left off
            next_chunk_id = max([chunk.get("id", -1) for chunk in existing_processed_chunks], default=-1) + 1
            
            # Create new processed chunks
            new_processed_chunks = []
            for i, chunk in enumerate(new_chunks):
                chunk_data = {
                    "id": next_chunk_id + i,
                    "original_text": chunk,
                    "current_text": chunk,
                    "spell_checked": False,
                    "grammar_checked": False,
                    "human_edited": False,
                    "char_count": len(chunk)
                }
                new_processed_chunks.append(chunk_data)
            
            # Combine existing and new chunks
            all_processed_chunks = existing_processed_chunks + new_processed_chunks
            
            return {
                **state,
                "processed_chunks": all_processed_chunks,
                "current_step": "ready_for_tts",
                "error_message": None
            }
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            
            # Provide detailed error information
            extracted_texts = state.get("extracted_texts", [])
            text_sources = [f"Page {item.get('page', '?')}: {item.get('filename', 'unknown')}" 
                          for item in extracted_texts]
            
            detailed_error = f"Text processing failed for {len(extracted_texts)} extracted texts ({', '.join(text_sources[:3])}{'...' if len(text_sources) > 3 else ''}). Error: {str(e)}"
            
            return {
                **state,
                "error_message": detailed_error,
                "error_details": error_details,
                "failed_texts": text_sources,
                "current_step": "error"
            }
    
    def process_chunk_spell_check(self, state: TextReaderState, chunk_id: int, api_key: str = None,
                                ai_provider: str = "OpenAI", ollama_url: str = None, ollama_model: str = None, timeout: int = 90) -> TextReaderState:
        """Perform spell check on a specific chunk using specified AI provider"""
        try:
            # Use provided api_key or fall back to state api_key
            if not api_key:
                api_key = state.get("api_key")
            
            # Check provider requirements
            if ai_provider == "OpenAI" and not api_key:
                raise Exception("OpenAI API key required for spell check with OpenAI")
            elif ai_provider == "Ollama" and (not ollama_url or not ollama_model):
                raise Exception("Ollama URL and model required for spell check with Ollama")
            
            processed_chunks = state.get("processed_chunks", [])
            if chunk_id >= len(processed_chunks):
                raise Exception("Invalid chunk ID")
            
            chunk = processed_chunks[chunk_id]
            original_text = chunk["current_text"]
            
            corrected_text = self.spell_check_with_ai(original_text, api_key, ai_provider, ollama_url, ollama_model, timeout)
            
            # Update the chunk
            processed_chunks[chunk_id]["current_text"] = corrected_text
            processed_chunks[chunk_id]["spell_checked"] = True
            
            return {
                **state,
                "processed_chunks": processed_chunks,
                "error_message": None
            }
            
        except Exception as e:
            return {
                **state,
                "error_message": f"Spell check error: {str(e)}"
            }
    
    def process_chunk_grammar_check(self, state: TextReaderState, chunk_id: int) -> TextReaderState:
        """Perform grammar check on a specific chunk"""
        try:
            api_key = state.get("api_key")
            if not api_key:
                raise Exception("OpenAI API key required for grammar check")
            
            processed_chunks = state.get("processed_chunks", [])
            if chunk_id >= len(processed_chunks):
                raise Exception("Invalid chunk ID")
            
            chunk = processed_chunks[chunk_id]
            original_text = chunk["current_text"]
            
            corrected_text = self.grammar_check_openai(original_text, api_key)
            
            # Update the chunk
            processed_chunks[chunk_id]["current_text"] = corrected_text
            processed_chunks[chunk_id]["grammar_checked"] = True
            
            return {
                **state,
                "processed_chunks": processed_chunks,
                "error_message": None
            }
            
        except Exception as e:
            return {
                **state,
                "error_message": f"Grammar check error: {str(e)}"
            }
    
    def process_chunk_human_edit(self, state: TextReaderState, chunk_id: int, edited_text: str) -> TextReaderState:
        """Apply human edits to a specific chunk"""
        try:
            processed_chunks = state.get("processed_chunks", [])
            if chunk_id >= len(processed_chunks):
                raise Exception("Invalid chunk ID")
            
            # Store the previous version for potential undo
            chunk = processed_chunks[chunk_id]
            previous_text = chunk["current_text"]
            
            # Update the chunk with human edits
            processed_chunks[chunk_id]["current_text"] = edited_text
            processed_chunks[chunk_id]["human_edited"] = True
            processed_chunks[chunk_id]["previous_text"] = previous_text
            processed_chunks[chunk_id]["edit_timestamp"] = datetime.now().isoformat()
            processed_chunks[chunk_id]["char_count"] = len(edited_text)
            
            return {
                **state,
                "processed_chunks": processed_chunks,
                "error_message": None
            }
            
        except Exception as e:
            return {
                **state,
                "error_message": f"Human edit error: {str(e)}"
            }
    
    def undo_chunk_edit(self, state: TextReaderState, chunk_id: int) -> TextReaderState:
        """Undo the last edit on a specific chunk"""
        try:
            processed_chunks = state.get("processed_chunks", [])
            if chunk_id >= len(processed_chunks):
                raise Exception("Invalid chunk ID")
            
            chunk = processed_chunks[chunk_id]
            if "previous_text" not in chunk:
                raise Exception("No previous version available to undo")
            
            # Restore the previous version
            previous_text = chunk["previous_text"]
            processed_chunks[chunk_id]["current_text"] = previous_text
            processed_chunks[chunk_id]["char_count"] = len(previous_text)
            
            # Remove undo information
            del processed_chunks[chunk_id]["previous_text"]
            if "edit_timestamp" in processed_chunks[chunk_id]:
                del processed_chunks[chunk_id]["edit_timestamp"]
            processed_chunks[chunk_id]["human_edited"] = False
            
            return {
                **state,
                "processed_chunks": processed_chunks,
                "error_message": None
            }
            
        except Exception as e:
            return {
                **state,
                "error_message": f"Undo edit error: {str(e)}"
            }
    
    def get_chunk_info(self, state: TextReaderState, chunk_id: int) -> Dict[str, Any]:
        """Get detailed information about a specific chunk"""
        try:
            processed_chunks = state.get("processed_chunks", [])
            if chunk_id >= len(processed_chunks):
                raise Exception("Invalid chunk ID")
            
            chunk = processed_chunks[chunk_id]
            
            return {
                "chunk_id": chunk_id,
                "original_text": chunk.get("original_text", ""),
                "current_text": chunk.get("current_text", ""),
                "char_count": chunk.get("char_count", 0),
                "word_count": len(chunk.get("current_text", "").split()),
                "spell_checked": chunk.get("spell_checked", False),
                "grammar_checked": chunk.get("grammar_checked", False),
                "human_edited": chunk.get("human_edited", False),
                "has_previous_version": "previous_text" in chunk,
                "edit_timestamp": chunk.get("edit_timestamp", None)
            }
            
        except Exception as e:
            return {"error": f"Get chunk info error: {str(e)}"}
    
    def get_all_chunks_info(self, state: TextReaderState) -> List[Dict[str, Any]]:
        """Get information about all chunks"""
        try:
            processed_chunks = state.get("processed_chunks", [])
            chunks_info = []
            
            for i in range(len(processed_chunks)):
                chunk_info = self.get_chunk_info(state, i)
                chunks_info.append(chunk_info)
            
            return chunks_info
            
        except Exception as e:
            return [{"error": f"Get all chunks info error: {str(e)}"}]

class ParallelEditingWindowAgent:
    """Agent responsible for managing parallel editing windows with grammar analysis"""
    
    def __init__(self):
        self.name = "parallel_editor"
        self.grammar_analyzer = GrammarAnalysisAgent()
    
    def create_parallel_editing_session(self, state: TextReaderState, chunk_id: int) -> Dict[str, Any]:
        """Create a parallel editing session for a specific chunk"""
        try:
            processed_chunks = state.get("processed_chunks", [])
            if chunk_id >= len(processed_chunks):
                raise Exception("Invalid chunk ID")
            
            chunk = processed_chunks[chunk_id]
            original_text = chunk["current_text"]
            
            # Analyze grammar and completeness
            analysis_state = self.grammar_analyzer.process_text_completeness(state, chunk_id)
            updated_chunk = analysis_state["processed_chunks"][chunk_id]
            
            editing_session = {
                "chunk_id": chunk_id,
                "original_text": original_text,
                "current_text": original_text,
                "parallel_text": original_text,  # Editable version
                "grammar_analysis": updated_chunk.get("grammar_analysis", {}),
                "completion_suggestions": updated_chunk.get("completion_suggestions", []),
                "needs_completion": updated_chunk.get("needs_completion", False),
                "session_id": f"edit_session_{chunk_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "created_at": datetime.now().isoformat(),
                "approved": False,
                "changes_made": False
            }
            
            return {
                "success": True,
                "editing_session": editing_session,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "editing_session": None,
                "error": f"Failed to create editing session: {str(e)}"
            }
    
    def update_parallel_text(self, editing_session: Dict[str, Any], new_text: str) -> Dict[str, Any]:
        """Update the parallel text in an editing session"""
        try:
            editing_session["parallel_text"] = new_text
            editing_session["changes_made"] = new_text != editing_session["original_text"]
            editing_session["last_modified"] = datetime.now().isoformat()
            
            return {
                "success": True,
                "updated_session": editing_session,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "updated_session": editing_session,
                "error": f"Failed to update parallel text: {str(e)}"
            }
    
    def approve_parallel_edits(self, state: TextReaderState, editing_session: Dict[str, Any]) -> TextReaderState:
        """Approve and apply parallel edits to the main text"""
        try:
            chunk_id = editing_session["chunk_id"]
            parallel_text = editing_session["parallel_text"]
            
            processed_chunks = state.get("processed_chunks", [])
            if chunk_id >= len(processed_chunks):
                raise Exception("Invalid chunk ID")
            
            # Apply the parallel edits to the main chunk
            processed_chunks[chunk_id]["current_text"] = parallel_text
            processed_chunks[chunk_id]["human_edited"] = True
            processed_chunks[chunk_id]["parallel_editing_session"] = editing_session["session_id"]
            processed_chunks[chunk_id]["edit_timestamp"] = datetime.now().isoformat()
            processed_chunks[chunk_id]["char_count"] = len(parallel_text)
            
            # Mark editing session as approved
            editing_session["approved"] = True
            editing_session["approved_at"] = datetime.now().isoformat()
            
            return {
                **state,
                "processed_chunks": processed_chunks,
                "human_edited": True,
                "error_message": None
            }
            
        except Exception as e:
            return {
                **state,
                "error_message": f"Failed to approve parallel edits: {str(e)}"
            }
    
    def reject_parallel_edits(self, editing_session: Dict[str, Any]) -> Dict[str, Any]:
        """Reject parallel edits and revert to original"""
        editing_session["parallel_text"] = editing_session["original_text"]
        editing_session["changes_made"] = False
        editing_session["rejected_at"] = datetime.now().isoformat()
        
        return {
            "success": True,
            "updated_session": editing_session,
            "error": None
        }
    
    def apply_completion_suggestion(self, editing_session: Dict[str, Any], suggestion_index: int) -> Dict[str, Any]:
        """Apply a specific completion suggestion to the parallel text"""
        try:
            completion_suggestions = editing_session.get("completion_suggestions", [])
            if suggestion_index >= len(completion_suggestions):
                raise Exception("Invalid suggestion index")
            
            suggestion = completion_suggestions[suggestion_index]
            current_parallel = editing_session["parallel_text"]
            
            # Replace the original incomplete sentence with the completed version
            updated_text = current_parallel.replace(
                suggestion["original"], 
                suggestion["completed"]
            )
            
            editing_session["parallel_text"] = updated_text
            editing_session["changes_made"] = True
            editing_session["last_completion_applied"] = {
                "suggestion_index": suggestion_index,
                "original": suggestion["original"],
                "completed": suggestion["completed"],
                "applied_at": datetime.now().isoformat()
            }
            
            return {
                "success": True,
                "updated_session": editing_session,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "updated_session": editing_session,
                "error": f"Failed to apply completion suggestion: {str(e)}"
            }
    
    def improve_text_with_openai(self, editing_session: Dict[str, Any], api_key: str, improvement_type: str = "general") -> Dict[str, Any]:
        """Improve text using OpenAI with specified improvement type"""
        try:
            if not api_key:
                raise Exception("OpenAI API key required for text improvement")
            
            current_text = editing_session["parallel_text"]
            text_processor = TextProcessingAgent()
            
            # Use the new improve_text_openai method from TextProcessingAgent
            improved_text = text_processor.improve_text_openai(current_text, api_key, improvement_type)
            
            # Update the editing session
            editing_session["parallel_text"] = improved_text
            editing_session["changes_made"] = True
            editing_session["last_ai_improvement"] = {
                "improvement_type": improvement_type,
                "original_text": current_text,
                "improved_text": improved_text,
                "applied_at": datetime.now().isoformat()
            }
            
            return {
                "success": True,
                "updated_session": editing_session,
                "improved_text": improved_text,
                "improvement_type": improvement_type,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "updated_session": editing_session,
                "improved_text": None,
                "improvement_type": improvement_type,
                "error": f"Text improvement failed: {str(e)}"
            }
    
    def improve_text_with_ai(self, editing_session: Dict[str, Any], api_key: str = None, improvement_type: str = "general",
                           ai_provider: str = "OpenAI", ollama_url: str = None, ollama_model: str = None) -> Dict[str, Any]:
        """Improve text using AI (OpenAI or Ollama) with specified improvement type"""
        try:
            current_text = editing_session["parallel_text"]
            text_processor = TextProcessingAgent()
            
            # Use the new improve_text_with_ai method from TextProcessingAgent
            improved_text = text_processor.improve_text_with_ai(
                current_text, api_key, improvement_type, ai_provider, ollama_url, ollama_model
            )
            
            # Update the editing session
            editing_session["parallel_text"] = improved_text
            editing_session["changes_made"] = True
            editing_session["last_ai_improvement"] = {
                "improvement_type": improvement_type,
                "original_text": current_text,
                "improved_text": improved_text,
                "applied_at": datetime.now().isoformat(),
                "ai_provider": ai_provider
            }
            
            return {
                "success": True,
                "updated_session": editing_session,
                "improved_text": improved_text,
                "improvement_type": improvement_type,
                "ai_provider": ai_provider,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "updated_session": editing_session,
                "improved_text": None,
                "improvement_type": improvement_type,
                "ai_provider": ai_provider,
                "error": f"Text improvement failed: {str(e)}"
            }
    
    def summarize_text_with_openai(self, editing_session: Dict[str, Any], api_key: str, target_length: int = 350) -> Dict[str, Any]:
        """Summarize text using OpenAI with specific line references"""
        try:
            if not api_key:
                raise Exception("OpenAI API key required for text summarization")
            
            current_text = editing_session["parallel_text"]
            text_processor = TextProcessingAgent()
            
            # Use the new summarize_text_openai method from TextProcessingAgent
            summary_text = text_processor.summarize_text_openai(current_text, api_key, target_length)
            
            # Store the summary in the editing session
            editing_session["last_summary"] = {
                "summary_text": summary_text,
                "target_length": target_length,
                "source_text": current_text,
                "generated_at": datetime.now().isoformat()
            }
            
            return {
                "success": True,
                "updated_session": editing_session,
                "summary_text": summary_text,
                "target_length": target_length,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "updated_session": editing_session,
                "summary_text": None,
                "target_length": target_length,
                "error": f"Text summarization failed: {str(e)}"
            }
    
    def summarize_text_with_ai(self, editing_session: Dict[str, Any], api_key: str = None, target_length: int = 350,
                             ai_provider: str = "OpenAI", ollama_url: str = None, ollama_model: str = None, timeout: int = 180) -> Dict[str, Any]:
        """Summarize text using AI (OpenAI or Ollama) with specific line references"""
        try:
            current_text = editing_session["parallel_text"]
            text_processor = TextProcessingAgent()
            
            # Use the new summarize_text_with_ai method from TextProcessingAgent
            summary_text = text_processor.summarize_text_with_ai(
                current_text, api_key, target_length, ai_provider, ollama_url, ollama_model, timeout
            )
            
            # Store the summary in the editing session
            editing_session["last_summary"] = {
                "summary_text": summary_text,
                "target_length": target_length,
                "source_text": current_text,
                "generated_at": datetime.now().isoformat(),
                "ai_provider": ai_provider
            }
            
            return {
                "success": True,
                "updated_session": editing_session,
                "summary_text": summary_text,
                "target_length": target_length,
                "ai_provider": ai_provider,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "updated_session": editing_session,
                "summary_text": None,
                "target_length": target_length,
                "ai_provider": ai_provider,
                "error": f"Text summarization failed: {str(e)}"
            }
    
    def generate_speech_and_summary(self, editing_session: Dict[str, Any], api_key: str, 
                                   tts_engine: str = "Google TTS", tts_settings: Dict[str, Any] = None,
                                   include_speech: bool = True, include_summary: bool = True,
                                   summary_length: int = 350, base_filename: str = None,
                                   summarization_ai: str = "OpenAI", ollama_url: str = None, 
                                   ollama_model: str = None) -> Dict[str, Any]:
        """Generate speech and/or summary content and create HTML export"""
        try:
            if not include_speech and not include_summary:
                raise Exception("At least one option (speech or summary) must be enabled")
            
            current_text = editing_session["parallel_text"]
            if not current_text.strip():
                raise Exception("No text content to process")
            
            result_data = {
                "text_content": current_text,
                "audio_data": None,
                "audio_info": None,
                "summary_text": None,
                "html_content": None,
                "generated_at": datetime.now().isoformat()
            }
            
            # Generate speech if requested
            if include_speech:
                if tts_settings is None:
                    tts_settings = {}
                
                # Use TTS agent functionality
                tts_agent = TextToSpeechAgent()
                
                if tts_engine == "OpenAI TTS (Premium)":
                    if not api_key:
                        raise Exception("OpenAI API key required for OpenAI TTS")
                    
                    voice = tts_settings.get("voice", "alloy")
                    model = tts_settings.get("model", "tts-1-hd")
                    
                    audio_bytes_io = tts_agent.text_to_speech_openai(current_text, api_key, voice, model)
                    audio_data = audio_bytes_io.getvalue()
                    
                    result_data["audio_data"] = audio_data
                    result_data["audio_info"] = {
                        "engine": "OpenAI TTS",
                        "voice": voice,
                        "model": model,
                        "text_length": len(current_text),
                        "word_count": len(current_text.split())
                    }
                else:
                    # Google TTS
                    language = tts_settings.get("language", "en")
                    audio_bytes_io = tts_agent.text_to_speech_gtts(current_text, language)
                    audio_data = audio_bytes_io.getvalue()
                    
                    result_data["audio_data"] = audio_data
                    result_data["audio_info"] = {
                        "engine": "Google TTS",
                        "language": language,
                        "text_length": len(current_text),
                        "word_count": len(current_text.split())
                    }
            
            # Generate summary if requested
            if include_summary:
                # Check AI provider requirements
                if summarization_ai == "OpenAI" and not api_key:
                    raise Exception("OpenAI API key required for text summarization with OpenAI")
                elif summarization_ai == "Ollama" and (not ollama_url or not ollama_model):
                    raise Exception("Ollama URL and model required for text summarization with Ollama")
                
                text_processor = TextProcessingAgent()
                # Use timeout for summarization, default to 180 seconds if not provided 
                summary_timeout = getattr(st.session_state, 'current_ollama_timeout', 180) if summarization_ai == "Ollama" else 180
                summary_text = text_processor.summarize_text_with_ai(
                    current_text, api_key, summary_length, summarization_ai, ollama_url, ollama_model, summary_timeout
                )
                result_data["summary_text"] = summary_text
                
                # Store AI model information for HTML export
                if summarization_ai == "OpenAI":
                    result_data["summary_model"] = "OpenAI GPT-3.5 Turbo"
                elif summarization_ai == "Ollama":
                    result_data["summary_model"] = f"Ollama {ollama_model}"
                else:
                    result_data["summary_model"] = summarization_ai
            
            # Generate HTML content
            result_data["html_content"] = self._create_html_export(result_data, editing_session, base_filename)
            result_data["base_filename"] = base_filename  # Store for UI to use in download names
            
            # Update editing session with the combined results
            editing_session["last_combined_export"] = result_data
            
            return {
                "success": True,
                "updated_session": editing_session,
                "export_data": result_data,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "updated_session": editing_session,
                "export_data": None,
                "error": f"Combined generation failed: {str(e)}"
            }
    
    def _create_html_export(self, result_data: Dict[str, Any], editing_session: Dict[str, Any], base_filename: str = None) -> str:
        """Create HTML content for export with embedded audio, summary, and synchronized text highlighting"""
        import base64
        import re
        
        # Get metadata
        generated_at = datetime.fromisoformat(result_data["generated_at"]).strftime("%Y-%m-%d %H:%M:%S")
        text_content = result_data["text_content"]
        summary_text = result_data.get("summary_text")
        audio_data = result_data.get("audio_data")
        audio_info = result_data.get("audio_info")
        
        # Prepare synchronized text if audio is available
        synchronized_text_html = self._create_synchronized_text(text_content) if audio_data else text_content
        
        # Create HTML structure with synchronized highlighting
        title = base_filename if base_filename else f"Text Processing Export - {generated_at}"
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 3px solid #007bff;
        }}
        .header h1 {{
            color: #007bff;
            margin: 0;
        }}
        .metadata {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 25px;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .section-title {{
            color: #495057;
            border-bottom: 2px solid #dee2e6;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}
        .original-text {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            border-left: 4px solid #28a745;
            white-space: pre-wrap;
            font-family: 'Georgia', serif;
            font-size: 18px;
            line-height: 1.8;
        }}
        .sync-text {{
            transition: all 0.3s ease;
            padding: 2px 4px;
            border-radius: 3px;
            cursor: pointer;
        }}
        .sync-text.current {{
            background-color: #ffd700;
            color: #333;
            font-weight: bold;
            box-shadow: 0 2px 4px rgba(255, 215, 0, 0.3);
        }}
        .sync-text.previous {{
            background-color: #e8f5e8;
            color: #4a5568;
        }}
        .sync-text:hover {{
            background-color: #e6f3ff;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        }}
        .audio-controls {{
            background-color: #fff3e0;
            padding: 20px;
            border-radius: 5px;
            border-left: 4px solid #ff9800;
            text-align: center;
            margin-bottom: 20px;
        }}
        .control-buttons {{
            margin: 15px 0;
        }}
        .control-button {{
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 0 5px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }}
        .control-button:hover {{
            background-color: #0056b3;
        }}
        .control-button:disabled {{
            background-color: #6c757d;
            cursor: not-allowed;
        }}
        .progress-bar {{
            width: 100%;
            height: 8px;
            background-color: #e9ecef;
            border-radius: 4px;
            margin: 15px 0;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background-color: #007bff;
            border-radius: 4px;
            width: 0%;
            transition: width 0.1s ease;
        }}
        .time-display {{
            color: #6c757d;
            font-family: monospace;
            margin: 10px 0;
        }}
        .summary-text {{
            background-color: #e3f2fd;
            padding: 20px;
            border-radius: 5px;
            border-left: 4px solid #2196f3;
        }}
        .audio-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .audio-info-item {{
            background-color: white;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }}
        .footer {{
            margin-top: 40px;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
        }}
        audio {{
            width: 100%;
            max-width: 500px;
            margin: 15px 0;
        }}
        .instructions {{
            background-color: #d1ecf1;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #17a2b8;
            margin-bottom: 20px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📖 Interactive Text Reader</h1>
            <p>Generated using LangGraph Multi-Agent System with Synchronized Highlighting</p>
        </div>
        
        <div class="metadata">
            <strong>📅 Generated:</strong> {generated_at}<br>
            <strong>📄 Text Length:</strong> {len(text_content)} characters ({len(text_content.split())} words)<br>
            <strong>🔧 Processing:</strong> {"Speech + Summary + Interactive Sync" if summary_text and audio_data else "Speech + Interactive Sync" if audio_data else "Summary Only"}
        </div>"""
        
        # Add interactive text section with audio controls
        if audio_data:
            html_content += f"""
        <div class="instructions">
            <strong>🎵 Interactive Features:</strong> Click play to start synchronized highlighting. Click any sentence to jump to that position. Use keyboard shortcuts: Space = Play/Pause, ← → = Skip 10s.
        </div>
        
        <div class="section">
            <h2 class="section-title">🎵 Interactive Text with Synchronized Audio</h2>
            <div class="audio-controls">
                <p><strong>🎤 Text-to-Speech Player</strong></p>
                <div class="control-buttons">
                    <button class="control-button" id="playPauseBtn" onclick="togglePlayPause()">▶️ Play</button>
                    <button class="control-button" id="rewindBtn" onclick="skipTime(-10)">⏪ -10s</button>
                    <button class="control-button" id="forwardBtn" onclick="skipTime(10)">⏩ +10s</button>
                    <button class="control-button" id="restartBtn" onclick="restartAudio()">🔄 Restart</button>
                </div>
                <div class="progress-bar" onclick="seekToPosition(event)">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <div class="time-display">
                    <span id="currentTime">0:00</span> / <span id="totalTime">0:00</span>
                </div>
                <audio id="audioPlayer" preload="auto">
                    <source src="data:audio/mp3;base64,{base64.b64encode(audio_data).decode()}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
            </div>
            <div class="original-text" id="textContainer">
                {synchronized_text_html}
            </div>
        </div>"""
        else:
            html_content += f"""
        <div class="section">
            <h2 class="section-title">📄 Original Text</h2>
            <div class="original-text">{text_content}</div>
        </div>"""
        
        # Add summary section if available
        if summary_text:
            summary_word_count = len(summary_text.split())
            formatted_summary = summary_text.replace('\n', '<br>')
            summary_model = result_data.get("summary_model", "Unknown AI Model")
            html_content += f"""
        <div class="section">
            <h2 class="section-title">📋 AI-Generated Summary ({summary_word_count} words)</h2>
            <div class="model-attribution">
                <small style="color: #666; font-style: italic;">Generated by: {summary_model}</small>
            </div>
            <div class="summary-text">
                {formatted_summary}
            </div>
        </div>"""
        
        # Add audio section if available
        if audio_data:
            # Encode audio as base64 for embedding
            audio_b64 = base64.b64encode(audio_data).decode()
            audio_src = f"data:audio/mp3;base64,{audio_b64}"
            
            html_content += f"""
        <div class="section">
            <h2 class="section-title">🎵 Generated Audio</h2>
            <div class="audio-section">
                <p><strong>🎤 Text-to-Speech Audio</strong></p>
                <audio controls>
                    <source src="{audio_src}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
                <div class="audio-info">
                    <div class="audio-info-item">
                        <strong>Engine</strong><br>{audio_info['engine']}
                    </div>"""
            
            if 'voice' in audio_info:
                html_content += f"""
                    <div class="audio-info-item">
                        <strong>Voice</strong><br>{audio_info['voice']}
                    </div>"""
            
            if 'language' in audio_info:
                html_content += f"""
                    <div class="audio-info-item">
                        <strong>Language</strong><br>{audio_info['language']}
                    </div>"""
            
            if 'model' in audio_info:
                html_content += f"""
                    <div class="audio-info-item">
                        <strong>Model</strong><br>{audio_info['model']}
                    </div>"""
            
            html_content += f"""
                    <div class="audio-info-item">
                        <strong>Duration Est.</strong><br>{audio_info['word_count'] / 150:.1f} min
                    </div>
                </div>
            </div>
        </div>"""
        
        # Add footer
        html_content += f"""
        <div class="footer">
            <p>🤖 Generated with <strong>LangGraph Multi-Agent Text Reader</strong></p>
            <p>Powered by OpenAI GPT-3.5 Turbo • Text-to-Speech • Advanced Grammar Analysis</p>
        </div>
    </div>

    <!-- JavaScript for synchronized text highlighting -->
    <script>
        let audioPlayer = null;
        let syncSpans = [];
        let currentSentence = -1;
        let isPlaying = false;
        let animationFrame = null;

        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', function() {{
            audioPlayer = document.getElementById('audioPlayer');
            syncSpans = Array.from(document.querySelectorAll('.sync-text'));
            
            if (audioPlayer && syncSpans.length > 0) {{
                setupAudioPlayer();
                setupTextInteraction();
                setupKeyboardShortcuts();
            }}
        }});

        function setupAudioPlayer() {{
            // Set up audio player events
            audioPlayer.addEventListener('loadedmetadata', function() {{
                document.getElementById('totalTime').textContent = formatTime(audioPlayer.duration);
            }});

            audioPlayer.addEventListener('timeupdate', function() {{
                updateProgress();
                updateTextHighlight();
            }});

            audioPlayer.addEventListener('ended', function() {{
                resetPlayback();
            }});

            audioPlayer.addEventListener('play', function() {{
                isPlaying = true;
                document.getElementById('playPauseBtn').innerHTML = '⏸️ Pause';
                startProgressAnimation();
            }});

            audioPlayer.addEventListener('pause', function() {{
                isPlaying = false;
                document.getElementById('playPauseBtn').innerHTML = '▶️ Play';
                stopProgressAnimation();
            }});
        }}

        function setupTextInteraction() {{
            // Make text spans clickable for navigation
            syncSpans.forEach((span, index) => {{
                span.addEventListener('click', function() {{
                    const startTime = parseFloat(span.dataset.start);
                    if (audioPlayer && !isNaN(startTime)) {{
                        audioPlayer.currentTime = startTime;
                        highlightSentence(index);
                    }}
                }});
            }});
        }}

        function setupKeyboardShortcuts() {{
            document.addEventListener('keydown', function(e) {{
                // Prevent shortcuts when user is typing in input fields
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
                
                switch(e.code) {{
                    case 'Space':
                        e.preventDefault();
                        togglePlayPause();
                        break;
                    case 'ArrowLeft':
                        e.preventDefault();
                        skipTime(-10);
                        break;
                    case 'ArrowRight':
                        e.preventDefault();
                        skipTime(10);
                        break;
                    case 'KeyR':
                        e.preventDefault();
                        restartAudio();
                        break;
                }}
            }});
        }}

        function togglePlayPause() {{
            if (!audioPlayer) return;
            
            if (audioPlayer.paused) {{
                audioPlayer.play().catch(e => console.log('Playback failed:', e));
            }} else {{
                audioPlayer.pause();
            }}
        }}

        function skipTime(seconds) {{
            if (!audioPlayer) return;
            
            audioPlayer.currentTime = Math.max(0, Math.min(audioPlayer.duration, audioPlayer.currentTime + seconds));
        }}

        function restartAudio() {{
            if (!audioPlayer) return;
            
            audioPlayer.currentTime = 0;
            currentSentence = -1;
            clearAllHighlights();
        }}

        function seekToPosition(event) {{
            if (!audioPlayer) return;
            
            const progressBar = event.currentTarget;
            const rect = progressBar.getBoundingClientRect();
            const clickX = event.clientX - rect.left;
            const percentage = clickX / rect.width;
            const newTime = percentage * audioPlayer.duration;
            
            audioPlayer.currentTime = newTime;
        }}

        function updateProgress() {{
            if (!audioPlayer) return;
            
            const currentTime = audioPlayer.currentTime;
            const duration = audioPlayer.duration;
            const percentage = (currentTime / duration) * 100;
            
            document.getElementById('progressFill').style.width = percentage + '%';
            document.getElementById('currentTime').textContent = formatTime(currentTime);
        }}

        function updateTextHighlight() {{
            if (!audioPlayer || syncSpans.length === 0) return;
            
            const currentTime = audioPlayer.currentTime;
            let newCurrentSentence = -1;
            
            // Find which sentence should be highlighted
            for (let i = 0; i < syncSpans.length; i++) {{
                const span = syncSpans[i];
                const startTime = parseFloat(span.dataset.start);
                const endTime = parseFloat(span.dataset.end);
                
                if (currentTime >= startTime && currentTime < endTime) {{
                    newCurrentSentence = i;
                    break;
                }}
            }}
            
            // Update highlighting if sentence changed
            if (newCurrentSentence !== currentSentence) {{
                highlightSentence(newCurrentSentence);
                currentSentence = newCurrentSentence;
            }}
        }}

        function highlightSentence(index) {{
            // Clear all highlights
            clearAllHighlights();
            
            if (index >= 0 && index < syncSpans.length) {{
                // Highlight current sentence
                syncSpans[index].classList.add('current');
                
                // Mark previous sentences as read
                for (let i = 0; i < index; i++) {{
                    syncSpans[i].classList.add('previous');
                }}
                
                // Scroll current sentence into view
                syncSpans[index].scrollIntoView({{
                    behavior: 'smooth',
                    block: 'center'
                }});
            }}
        }}

        function clearAllHighlights() {{
            syncSpans.forEach(span => {{
                span.classList.remove('current', 'previous');
            }});
        }}

        function resetPlayback() {{
            isPlaying = false;
            document.getElementById('playPauseBtn').innerHTML = '▶️ Play';
            clearAllHighlights();
            currentSentence = -1;
            stopProgressAnimation();
        }}

        function startProgressAnimation() {{
            function animate() {{
                if (isPlaying && audioPlayer && !audioPlayer.paused) {{
                    updateProgress();
                    animationFrame = requestAnimationFrame(animate);
                }}
            }}
            animationFrame = requestAnimationFrame(animate);
        }}

        function stopProgressAnimation() {{
            if (animationFrame) {{
                cancelAnimationFrame(animationFrame);
                animationFrame = null;
            }}
        }}

        function formatTime(seconds) {{
            if (isNaN(seconds)) return '0:00';
            
            const minutes = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return minutes + ':' + (secs < 10 ? '0' : '') + secs;
        }}

        // Auto-play functionality (optional - can be removed if not desired)
        // Uncomment the next line to auto-play when page loads
        // window.addEventListener('load', () => setTimeout(() => togglePlayPause(), 1000));
    </script>
</body>
</html>"""
        
        return html_content
    
    def _create_synchronized_text(self, text_content: str) -> str:
        """Create HTML with time-synchronized spans for text highlighting"""
        import re
        
        # Split text into sentences for better synchronization
        sentences = re.split(r'([.!?]+)', text_content)
        
        # Estimate reading speed (average 150 words per minute = 2.5 words per second)
        words_per_second = 2.5
        
        synchronized_html = ""
        current_time = 0.0
        
        sentence_index = 0
        for i in range(0, len(sentences), 2):  # Process sentence + punctuation pairs
            if i < len(sentences):
                sentence_text = sentences[i].strip()
                punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
                
                if sentence_text:  # Skip empty sentences
                    # Calculate duration based on word count
                    word_count = len(sentence_text.split())
                    duration = word_count / words_per_second if word_count > 0 else 1.0
                    
                    # Create time-synced span
                    synchronized_html += f'''<span class="sync-text" data-start="{current_time:.1f}" data-end="{current_time + duration:.1f}" data-sentence="{sentence_index}">{sentence_text}{punctuation}</span> '''
                    
                    current_time += duration
                    sentence_index += 1
                else:
                    # Handle punctuation-only or whitespace
                    synchronized_html += punctuation + " "
        
        return synchronized_html.strip()
    
    def get_editing_comparison(self, editing_session: Dict[str, Any]) -> Dict[str, Any]:
        """Get a comparison between original and parallel texts"""
        try:
            import difflib
            
            original_lines = editing_session["original_text"].splitlines()
            parallel_lines = editing_session["parallel_text"].splitlines()
            
            # Generate diff
            diff = list(difflib.unified_diff(
                original_lines, 
                parallel_lines, 
                fromfile='Original', 
                tofile='Edited', 
                lineterm=''
            ))
            
            # Calculate statistics
            total_original_chars = len(editing_session["original_text"])
            total_parallel_chars = len(editing_session["parallel_text"])
            char_difference = total_parallel_chars - total_original_chars
            
            return {
                "original_text": editing_session["original_text"],
                "parallel_text": editing_session["parallel_text"],
                "diff": diff,
                "statistics": {
                    "original_chars": total_original_chars,
                    "parallel_chars": total_parallel_chars,
                    "char_difference": char_difference,
                    "changes_made": editing_session.get("changes_made", False)
                },
                "grammar_analysis": editing_session.get("grammar_analysis", {}),
                "completion_suggestions": editing_session.get("completion_suggestions", [])
            }
            
        except Exception as e:
            return {
                "error": f"Failed to generate comparison: {str(e)}"
            }

class HumanEditingAgent:
    """Agent responsible for managing human editing interactions"""
    
    def __init__(self):
        self.name = "human_editor"
        self.text_processor = TextProcessingAgent()
        self.parallel_editor = ParallelEditingWindowAgent()
    
    def initiate_editing_mode(self, state: TextReaderState) -> TextReaderState:
        """Set the workflow into editing mode"""
        return {
            **state,
            "editing_mode": True,
            "current_step": "human_editing"
        }
    
    def exit_editing_mode(self, state: TextReaderState) -> TextReaderState:
        """Exit editing mode and proceed to TTS conversion"""
        return {
            **state,
            "editing_mode": False,
            "current_step": "ready_for_tts"
        }
    
    def create_parallel_editing_session(self, state: TextReaderState, chunk_id: int) -> Dict[str, Any]:
        """Create a parallel editing session using the parallel editor"""
        return self.parallel_editor.create_parallel_editing_session(state, chunk_id)
    
    def approve_parallel_edits(self, state: TextReaderState, editing_session: Dict[str, Any]) -> TextReaderState:
        """Approve and apply parallel edits"""
        return self.parallel_editor.approve_parallel_edits(state, editing_session)
    
    def generate_speech_from_parallel_edit(self, state: TextReaderState, editing_session: Dict[str, Any], 
                                         tts_engine: str = "Google TTS", tts_settings: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate speech from parallel edited text"""
        try:
            if tts_settings is None:
                tts_settings = {}
            
            text_to_speak = editing_session["parallel_text"]
            chunk_id = editing_session["chunk_id"]
            
            # Use the TTS agent to generate speech
            tts_agent = TextToSpeechAgent()
            
            # Temporarily update state with TTS settings
            temp_state = {
                **state,
                "tts_engine": tts_engine,
                "tts_settings": tts_settings
            }
            
            # Convert the edited text to speech
            result_state = tts_agent.convert_chunk(temp_state, chunk_id)
            
            if result_state.get("error_message"):
                return {
                    "success": False,
                    "error": result_state["error_message"],
                    "audio_data": None
                }
            
            # Extract the generated audio
            audio_files = result_state.get("audio_files", [])
            if audio_files:
                latest_audio = audio_files[-1]  # Get the most recent audio file
                return {
                    "success": True,
                    "error": None,
                    "audio_data": latest_audio["audio_data"],
                    "audio_info": {
                        "engine": latest_audio["engine"],
                        "settings": latest_audio["settings"],
                        "text_length": len(text_to_speak),
                        "word_count": len(text_to_speak.split()),
                        "session_id": editing_session["session_id"]
                    }
                }
            else:
                return {
                    "success": False,
                    "error": "No audio data generated",
                    "audio_data": None
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Speech generation error: {str(e)}",
                "audio_data": None
            }
    
    def edit_chunk(self, state: TextReaderState, chunk_id: int, edited_text: str) -> TextReaderState:
        """Apply human edits to a chunk"""
        result_state = self.text_processor.process_chunk_human_edit(state, chunk_id, edited_text)
        
        # Update global human_edited flag if edit was successful
        if not result_state.get("error_message"):
            result_state["human_edited"] = True
        
        return result_state
    
    def undo_edit(self, state: TextReaderState, chunk_id: int) -> TextReaderState:
        """Undo edits on a chunk"""
        result_state = self.text_processor.undo_chunk_edit(state, chunk_id)
        
        # Check if any chunks are still human edited
        if not result_state.get("error_message"):
            processed_chunks = result_state.get("processed_chunks", [])
            any_human_edited = any(chunk.get("human_edited", False) for chunk in processed_chunks)
            result_state["human_edited"] = any_human_edited
        
        return result_state
    
    def get_editing_summary(self, state: TextReaderState) -> Dict[str, Any]:
        """Get a summary of all editing activities"""
        try:
            processed_chunks = state.get("processed_chunks", [])
            
            total_chunks = len(processed_chunks)
            edited_chunks = sum(1 for chunk in processed_chunks if chunk.get("human_edited", False))
            spell_checked_chunks = sum(1 for chunk in processed_chunks if chunk.get("spell_checked", False))
            grammar_checked_chunks = sum(1 for chunk in processed_chunks if chunk.get("grammar_checked", False))
            
            edited_chunk_ids = [
                i for i, chunk in enumerate(processed_chunks) 
                if chunk.get("human_edited", False)
            ]
            
            return {
                "total_chunks": total_chunks,
                "edited_chunks": edited_chunks,
                "spell_checked_chunks": spell_checked_chunks,
                "grammar_checked_chunks": grammar_checked_chunks,
                "edited_chunk_ids": edited_chunk_ids,
                "editing_complete": not state.get("editing_mode", False)
            }
            
        except Exception as e:
            return {"error": f"Get editing summary error: {str(e)}"}
    
    def __call__(self, state: TextReaderState) -> TextReaderState:
        """Default behavior when called in workflow - initiate editing mode"""
        return self.initiate_editing_mode(state)

class TextToSpeechAgent:
    """Agent responsible for converting text to speech using various TTS engines"""
    
    def __init__(self):
        self.name = "tts_converter"
    
    def text_to_speech_openai(self, text: str, api_key: str, voice: str = "alloy", model: str = "tts-1-hd") -> io.BytesIO:
        """Convert text to speech using OpenAI TTS with rate limiting"""
        try:
            # OpenAI TTS has a 4096 character limit
            if len(text) > 4000:
                text = text[:4000]
            
            # Check rate limit before making request (both daily and per-minute)
            if not rate_limiter.can_make_request(1, text):
                usage_info = rate_limiter.get_usage_info()
                if usage_info["usage_percentage"] >= 100:
                    raise Exception(f"Daily API limit exceeded. Remaining requests: {usage_info['requests_remaining']}/{rate_limiter.daily_limit}")
                else:
                    raise Exception(f"Per-minute token limit would be exceeded. Current minute usage: {usage_info['tokens_current_minute']}/{rate_limiter.tokens_per_minute} tokens")
            
            client = OpenAI(api_key=api_key)
            response = client.audio.speech.create(
                model=model,
                voice=voice,
                input=text
            )
            
            # Record the successful request with text for token tracking
            rate_limiter.record_request(1, text)
            
            audio_bytes = io.BytesIO()
            for chunk in response.iter_bytes():
                audio_bytes.write(chunk)
            audio_bytes.seek(0)
            return audio_bytes
        except Exception as e:
            raise Exception(f"OpenAI TTS error: {str(e)}")
    
    def text_to_speech_gtts(self, text: str, language: str = 'en') -> io.BytesIO:
        """Convert text to speech using Google TTS"""
        try:
            tts = gTTS(text=text, lang=language, slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            return fp
        except Exception as e:
            raise Exception(f"Google TTS error: {str(e)}")
    
    def text_to_speech_mozilla(self, text: str, model: str = "tts_models/en/ljspeech/tacotron2-DDC", 
                             vocoder: str = "vocoder_models/en/ljspeech/hifigan_v2") -> io.BytesIO:
        """Convert text to speech using Mozilla TTS"""
        try:
            # Import Mozilla TTS
            from TTS.api import TTS
            import tempfile
            import os
            
            # Initialize TTS with model and vocoder
            tts = TTS(model_name=model, vocoder_name=vocoder)
            
            # Create a temporary file for output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Generate speech
            tts.tts_to_file(text=text, file_path=temp_path)
            
            # Read the generated audio file into BytesIO
            fp = io.BytesIO()
            with open(temp_path, 'rb') as audio_file:
                fp.write(audio_file.read())
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            fp.seek(0)
            return fp
            
        except ImportError as e:
            error_msg = "Mozilla/Coqui TTS not installed. Try these solutions:\n"
            error_msg += "1. pip install TTS --upgrade\n"
            error_msg += "2. pip install git+https://github.com/coqui-ai/TTS.git\n"
            error_msg += "3. conda install -c conda-forge tts\n"
            error_msg += "4. Check Python version (requires 3.8-3.11)\n"
            error_msg += f"Original error: {str(e)}"
            raise Exception(error_msg)
        except Exception as e:
            raise Exception(f"Mozilla TTS error: {str(e)}")
    
    def text_to_speech_coqui(self, text: str, model: str = "tts_models/multilingual/multi-dataset/xtts_v2",
                           speaker: str = None, language: str = "en", reference_audio: bytes = None) -> io.BytesIO:
        """Convert text to speech using Coqui TTS"""
        try:
            # Import Coqui TTS
            from TTS.api import TTS
            import tempfile
            import os
            
            # Initialize TTS with model
            tts = TTS(model_name=model)
            
            # Create a temporary file for output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Handle different model types
            if "xtts_v2" in model:
                # XTTS v2 supports voice cloning and multilingual
                if reference_audio and speaker == "custom":
                    # Voice cloning with uploaded audio
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_file:
                        ref_file.write(reference_audio)
                        ref_path = ref_file.name
                    
                    try:
                        tts.tts_to_file(
                            text=text,
                            file_path=temp_path,
                            speaker_wav=ref_path,
                            language=language
                        )
                    finally:
                        # Clean up reference file
                        os.unlink(ref_path)
                        
                elif speaker:
                    # Built-in speaker for XTTS v2
                    tts.tts_to_file(
                        text=text,
                        file_path=temp_path,
                        speaker=speaker,
                        language=language
                    )
                else:
                    # Default XTTS v2 generation
                    tts.tts_to_file(
                        text=text,
                        file_path=temp_path,
                        language=language
                    )
            else:
                # Standard TTS models
                if hasattr(tts.tts, 'speakers') and tts.tts.speakers:
                    # Multi-speaker model
                    speakers = tts.tts.speakers
                    selected_speaker = speaker if speaker in speakers else speakers[0]
                    tts.tts_to_file(text=text, file_path=temp_path, speaker=selected_speaker)
                else:
                    # Single speaker model
                    tts.tts_to_file(text=text, file_path=temp_path)
            
            # Read the generated audio file into BytesIO
            fp = io.BytesIO()
            with open(temp_path, 'rb') as audio_file:
                fp.write(audio_file.read())
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            fp.seek(0)
            return fp
            
        except ImportError as e:
            error_msg = "Coqui TTS not installed. Try these solutions:\n"
            error_msg += "1. pip install TTS --upgrade\n"
            error_msg += "2. pip install git+https://github.com/coqui-ai/TTS.git\n"
            error_msg += "3. conda install -c conda-forge tts\n"
            error_msg += "4. Check Python version (requires 3.8-3.11)\n"
            error_msg += f"Original error: {str(e)}"
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Coqui TTS error: {str(e)}\n"
            if "No module named" in str(e):
                error_msg += "This looks like an installation issue. Please check the installation guide."
            elif "CUDA" in str(e):
                error_msg += "GPU/CUDA issue detected. TTS will fall back to CPU processing."
            elif "model" in str(e).lower():
                error_msg += "Model loading failed. The model will be downloaded on first use."
            raise Exception(error_msg)
    
    def __call__(self, state: TextReaderState) -> TextReaderState:
        """Convert all chunks to speech for workflow usage"""
        return self.convert_all_chunks(state)
    
    def convert_chunk(self, state: TextReaderState, chunk_id: int) -> TextReaderState:
        """Convert a specific chunk to speech"""
        try:
            processed_chunks = state.get("processed_chunks", [])
            tts_engine = state.get("tts_engine", "Google TTS")
            tts_settings = state.get("tts_settings", {})
            api_key = state.get("api_key")
            
            if chunk_id >= len(processed_chunks):
                raise Exception("Invalid chunk ID")
            
            chunk = processed_chunks[chunk_id]
            text = chunk["current_text"]
            
            # Generate audio based on TTS engine
            if tts_engine == "OpenAI TTS (Premium)":
                if not api_key:
                    raise Exception("OpenAI API key required for OpenAI TTS")
                
                voice = tts_settings.get("voice", "alloy")
                model = tts_settings.get("model", "tts-1-hd")
                audio_bytes = self.text_to_speech_openai(text, api_key, voice, model)
            elif tts_engine == "Mozilla TTS (Local)":
                model = tts_settings.get("model", "tts_models/en/ljspeech/tacotron2-DDC")
                vocoder = tts_settings.get("vocoder", "vocoder_models/en/ljspeech/hifigan_v2")
                audio_bytes = self.text_to_speech_mozilla(text, model, vocoder)
            elif tts_engine == "Coqui TTS (Local)":
                model = tts_settings.get("model", "tts_models/multilingual/multi-dataset/xtts_v2")
                speaker = tts_settings.get("speaker")
                language = tts_settings.get("language", "en")
                reference_audio = tts_settings.get("reference_audio")  # bytes or None
                audio_bytes = self.text_to_speech_coqui(text, model, speaker, language, reference_audio)
            else:
                # Default to Google TTS
                language = tts_settings.get("language", "en")
                audio_bytes = self.text_to_speech_gtts(text, language)
            
            # Store audio file info
            audio_files = state.get("audio_files", [])
            audio_files.append({
                "chunk_id": chunk_id,
                "audio_data": audio_bytes.getvalue(),
                "engine": tts_engine,
                "settings": tts_settings
            })
            
            return {
                **state,
                "audio_files": audio_files,
                "error_message": None
            }
            
        except Exception as e:
            return {
                **state,
                "error_message": f"TTS conversion error: {str(e)}"
            }
    
    def convert_all_chunks(self, state: TextReaderState) -> TextReaderState:
        """Convert all chunks to speech"""
        import streamlit as st
        import time
        
        try:
            processed_chunks = state.get("processed_chunks", [])
            audio_files = []
            
            if not processed_chunks:
                return {
                    **state,
                    "error_message": "No processed chunks available for TTS conversion"
                }
            
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_text = st.empty()
            start_time = time.time()
            
            # Time estimates per TTS engine (seconds per 1000 characters)
            tts_engine = state.get("tts_engine", "Google TTS (Free)")
            time_estimates = {
                "Google TTS (Free)": 3,    # 3 seconds per 1000 chars
                "OpenAI TTS (Premium)": 5, # 5 seconds per 1000 chars
                "Mozilla TTS (Local)": 8,  # 8 seconds per 1000 chars
                "Coqui TTS (Local)": 10    # 10 seconds per 1000 chars
            }
            est_time_per_1k_chars = time_estimates.get(tts_engine, 5)
            
            for i, chunk in enumerate(processed_chunks):
                # Update progress
                progress = (i + 1) / len(processed_chunks)
                progress_bar.progress(progress)
                
                # Calculate estimates
                chunk_text = chunk.get("current_text", "")
                chunk_chars = len(chunk_text)
                estimated_chunk_time = (chunk_chars / 1000) * est_time_per_1k_chars
                
                # Update status
                status_text.text(f"🎵 Generating speech for chunk {i + 1} of {len(processed_chunks)} ({chunk_chars:,} chars)")
                
                # Calculate time estimates
                elapsed_time = time.time() - start_time
                if i > 0:
                    avg_time_per_chunk = elapsed_time / i
                    remaining_chunks = len(processed_chunks) - i
                    estimated_remaining = remaining_chunks * avg_time_per_chunk
                    time_text.text(f"⏱️ Elapsed: {elapsed_time:.1f}s | Estimated remaining: {estimated_remaining:.1f}s")
                else:
                    time_text.text(f"⏱️ Starting TTS conversion... Estimated: {estimated_chunk_time:.1f}s for this chunk")
                
                chunk_start_time = time.time()
                result_state = self.convert_chunk(state, i)
                chunk_time = time.time() - chunk_start_time
                
                if result_state.get("error_message"):
                    return result_state
                audio_files.extend(result_state.get("audio_files", []))
            
            # Complete progress tracking
            progress_bar.progress(1.0)
            total_time = time.time() - start_time
            status_text.text(f"✅ Completed generating speech for {len(processed_chunks)} chunks!")
            time_text.text(f"⏱️ Total time: {total_time:.1f}s | Average: {total_time/len(processed_chunks):.1f}s per chunk")
            
            return {
                **state,
                "audio_files": audio_files,
                "current_step": "completed",
                "error_message": None
            }
            
        except Exception as e:
            return {
                **state,
                "error_message": f"Batch TTS conversion error: {str(e)}"
            }

# Create the LangGraph workflow
def create_text_reader_workflow(include_human_editing: bool = True):
    """Create the main LangGraph workflow for the text reader application"""
    
    # Initialize agents
    file_processor = FileProcessingAgent()
    text_extractor = TextExtractionAgent()
    text_processor = TextProcessingAgent()
    human_editor = HumanEditingAgent()
    tts_agent = TextToSpeechAgent()
    
    # Create the state graph
    workflow = StateGraph(TextReaderState)
    
    # Add nodes
    workflow.add_node("file_processing", file_processor)
    workflow.add_node("text_extraction", text_extractor)
    workflow.add_node("text_processing", text_processor)
    
    if include_human_editing:
        workflow.add_node("human_editing", human_editor)
    
    workflow.add_node("tts_conversion", tts_agent)
    
    # Define the workflow edges
    workflow.set_entry_point("file_processing")
    workflow.add_edge("file_processing", "text_extraction")
    workflow.add_edge("text_extraction", "text_processing")
    
    if include_human_editing:
        workflow.add_edge("text_processing", "human_editing")
        workflow.add_edge("human_editing", "tts_conversion")
    else:
        workflow.add_edge("text_processing", "tts_conversion")
    
    workflow.add_edge("tts_conversion", END)
    
    # Compile the workflow
    app = workflow.compile()
    
    agents_dict = {
        "file_processor": file_processor,
        "text_extractor": text_extractor, 
        "text_processor": text_processor,
        "tts_agent": tts_agent
    }
    
    if include_human_editing:
        agents_dict["human_editor"] = human_editor
    
    return app, agents_dict

def create_text_reader_workflow_with_conditional_editing():
    """Create a workflow with conditional human editing based on state"""
    
    def should_edit(state: TextReaderState) -> str:
        """Conditional function to determine if human editing is needed"""
        editing_mode = state.get("editing_mode", False)
        if editing_mode:
            return "human_editing"
        else:
            return "tts_conversion"
    
    # Initialize agents
    file_processor = FileProcessingAgent()
    text_extractor = TextExtractionAgent()
    text_processor = TextProcessingAgent()
    human_editor = HumanEditingAgent()
    tts_agent = TextToSpeechAgent()
    
    # Create the state graph
    workflow = StateGraph(TextReaderState)
    
    # Add nodes
    workflow.add_node("file_processing", file_processor)
    workflow.add_node("text_extraction", text_extractor)
    workflow.add_node("text_processing", text_processor)
    workflow.add_node("human_editing", human_editor)
    workflow.add_node("tts_conversion", tts_agent)
    
    # Define the workflow edges
    workflow.set_entry_point("file_processing")
    workflow.add_edge("file_processing", "text_extraction")
    workflow.add_edge("text_extraction", "text_processing")
    
    # Add conditional edge for editing
    workflow.add_conditional_edges(
        "text_processing",
        should_edit,
        {
            "human_editing": "human_editing",
            "tts_conversion": "tts_conversion"
        }
    )
    
    workflow.add_edge("human_editing", "tts_conversion")
    workflow.add_edge("tts_conversion", END)
    
    # Compile the workflow
    app = workflow.compile()
    
    return app, {
        "file_processor": file_processor,
        "text_extractor": text_extractor, 
        "text_processor": text_processor,
        "human_editor": human_editor,
        "tts_agent": tts_agent
    }

def get_api_usage_info() -> Dict[str, Any]:
    """Get current API usage information"""
    return rate_limiter.get_usage_info()

def check_api_rate_limit(num_requests: int = 1, text: str = "") -> bool:
    """Check if we can make the specified number of API requests"""
    return rate_limiter.can_make_request(num_requests, text)

def estimate_requests_needed(chunks: List[Dict[str, Any]], operations: List[str]) -> int:
    """Estimate the number of API requests needed for the given operations"""
    request_count = 0
    
    for operation in operations:
        if operation in ["spell_check", "grammar_check", "tts_openai"]:
            request_count += len(chunks)
    
    return request_count