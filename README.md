# Robin Reader 📖🤖

A sophisticated multi-agent text processing system built with Streamlit and LangGraph that extracts text from documents, analyzes grammar, provides parallel editing capabilities, and converts text to interactive speech with synchronized highlighting.

## 🌟 Key Features

### 📄 **Document Processing**
- **PDF Text Extraction**: Advanced PDF parsing with text extraction
- **OCR Support**: Extract text from images (JPG, PNG, GIF) using Tesseract OCR or EasyOCR
- **Batch Processing**: Upload and process multiple files while preserving previous work
- **Smart Duplicate Detection**: Automatically skip previously processed files

### 🤖 **AI-Powered Multi-Agent System**
- **LangGraph Integration**: Sophisticated workflow orchestration with specialized agents
- **OpenAI GPT-3.5 Integration**: Advanced text processing and completion
- **Rate Limiting**: Smart API usage management (20,000 daily requests, token tracking)
- **Persistent Sessions**: Work accumulates across uploads without losing progress

### ✏️ **Advanced Grammar & Editing**
- **Grammar Analysis**: Detects incomplete sentences, fragments, run-on sentences
- **AI-Powered Completion**: GPT-3.5 completes incomplete sentences with context awareness
- **Parallel Editing Interface**: Side-by-side original and editable text with real-time diff
- **Human-in-the-Loop**: Review and approve AI suggestions before applying changes

### 🎵 **Interactive Text-to-Speech**
- **Multiple TTS Engines**: Google TTS (free), OpenAI TTS (premium), Mozilla TTS (local), and Coqui TTS (advanced local)
- **Voice Cloning**: Coqui TTS XTTS v2 supports custom voice cloning from audio samples
- **Multilingual Support**: Coqui TTS supports 14+ languages with natural-sounding voices
- **Synchronized Highlighting**: Real-time text highlighting during audio playbook
- **Interactive HTML Export**: Standalone files with embedded audio and click-to-navigate
- **Custom Audio Controls**: Play/pause, skip, restart, progress bar, keyboard shortcuts

### 📊 **Export & Summarization**
- **AI Text Summarization**: Generate detailed summaries with line references
- **Interactive HTML Files**: Complete reading experience with synchronized audio
- **Multiple Export Options**: Audio-only, summary-only, or combined exports
- **Offline Capability**: Exported HTML works without internet connection

## 🚀 Quick Start

### Installation

1. **Install Python 3.8+**
2. **Install OCR Engine:**
   ```bash
   # Windows - Download from https://github.com/UB-Mannheim/tesseract/wiki
   # macOS
   brew install tesseract
   # Linux
   sudo apt-get install tesseract-ocr
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

**Simple Version:**
```bash
streamlit run text_reader_app.py
```

**Advanced Multi-Agent Version:**
```bash
streamlit run text_reader_app_langgraph.py
```

**Grammar Checker Demo:**
```bash
streamlit run grammar_checker_demo.py
```

## 📋 Usage Workflow

### 1. **Document Upload & Processing**
- Upload PDF or image files via drag-and-drop interface
- Reorder files using up/down arrows
- Select OCR engine (Tesseract or EasyOCR)
- Process files with multi-agent system

### 2. **Grammar Analysis & Editing**
- View sentence completeness analysis with scoring
- Use parallel editing interface for side-by-side text modification
- Apply AI-powered completion suggestions for incomplete sentences
- Generate improved text with different styles (professional, casual, concise, etc.)

### 3. **Interactive Export**
- Generate speech with synchronized text highlighting
- Create detailed AI summaries with line references
- Export as interactive HTML with embedded audio
- Download individual components (audio, summary, HTML)

### 4. **Interactive Reading Experience**
- Open exported HTML file in any browser
- Click any sentence to jump to that audio position
- Use keyboard shortcuts (Space, Arrow keys, R)
- Watch text highlight in real-time during playback

## 🛠️ Architecture

### Multi-Agent System Components

- **FileProcessingAgent**: Validates uploads and manages file ordering
- **TextExtractionAgent**: PDF parsing and OCR with duplicate prevention
- **TextProcessingAgent**: Text chunking and preprocessing with session persistence
- **GrammarAnalysisAgent**: Advanced sentence analysis and LLM-powered completion
- **ParallelEditingWindowAgent**: Side-by-side editing with AI improvements
- **HumanEditingAgent**: Workflow orchestration and approval management
- **TextToSpeechAgent**: Multi-engine TTS with audio generation

### Technologies Used

- **Frontend**: Streamlit with custom UI components
- **AI/ML**: OpenAI GPT-3.5 Turbo, LangGraph workflow engine
- **OCR**: Tesseract OCR, EasyOCR
- **TTS**: Google Text-to-Speech (gTTS), OpenAI TTS, Mozilla TTS, Coqui TTS
- **Audio Processing**: PyAudio, base64 encoding for web embedding
- **Document Processing**: PyPDF2, Pillow for image handling

## 🎛️ Configuration Options

### TTS Engines
- **Google TTS**: Free, 10+ languages, good quality, requires internet
- **OpenAI TTS**: Premium, multiple voice options (alloy, echo, fable, onyx, nova, shimmer), requires API key
- **Mozilla TTS**: Local, high-quality neural voices, works offline, requires model download (~1-2GB)
- **Coqui TTS**: Advanced local TTS with voice cloning, XTTS v2 multilingual support, custom speakers

### OCR Engines
- **Tesseract**: Fast, lightweight, good for clean text
- **EasyOCR**: Better for complex layouts and handwriting

### Grammar Analysis
- **Completeness Detection**: Identifies missing subjects, predicates
- **Fragment Detection**: Finds sentence fragments and subordinating conjunctions
- **Run-on Detection**: Flags overly complex sentences
- **AI Completion**: Context-aware sentence completion with GPT-3.5

## 📁 Project Structure

```
robin-reader/
├── text_reader_app.py              # Simple single-threaded version
├── text_reader_app_langgraph.py    # Advanced multi-agent system
├── agents.py                       # Core agent implementations
├── parallel_editor_ui.py           # Parallel editing UI components
├── grammar_checker_demo.py         # Feature demonstration
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## 🔧 Advanced Features

### Session State Management
- **Persistent Work**: Accumulates results across multiple file uploads
- **State Recovery**: Automatic recovery from corrupted session states
- **Clear All Work**: Option to reset and start fresh

### Error Handling
- **Detailed Error Reporting**: File-specific error messages with troubleshooting
- **Graceful Fallbacks**: Automatic recovery from API failures
- **Debug Information**: Full stack traces and context for troubleshooting

### Interactive HTML Features
- **Synchronized Highlighting**: Gold highlight for current sentence, green for read
- **Click Navigation**: Click any sentence to jump to that audio position
- **Keyboard Controls**: Spacebar (play/pause), arrows (skip), R (restart)
- **Progress Tracking**: Visual progress bar with time display
- **Responsive Design**: Works on desktop and mobile devices

### Coqui TTS Advanced Features
- **XTTS v2 Voice Cloning**: Clone any voice from a 3-10 second audio sample
- **Built-in Speaker Library**: 12+ high-quality pre-trained speaker voices
- **Multilingual Generation**: Support for English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, and Japanese
- **Real-time Processing**: Fast neural voice synthesis with GPU acceleration (when available)
- **Custom Model Support**: Load and use custom-trained TTS models
- **Offline Operation**: Complete privacy with local processing, no internet required

## 🐛 Troubleshooting

### Common Issues

**OCR Problems:**
- Ensure Tesseract is in system PATH
- Try EasyOCR for better accuracy on complex images
- Check image quality and contrast

**Audio Issues:**
- Verify browser audio permissions
- Test with different browsers
- Check OpenAI API key for premium TTS
- For voice cloning: Use clear, 3-10 second audio samples in WAV/MP3 format

**TTS Installation Issues (Coqui/Mozilla TTS):**
- **Python Version**: Requires Python 3.8-3.11 (not 3.12+)
- **Method 1**: `pip install TTS --upgrade`
- **Method 2**: `pip install git+https://github.com/coqui-ai/TTS.git`
- **Method 3**: `pip install TTS[all] --upgrade`
- **Method 4**: `conda install -c conda-forge tts`
- **Windows**: Install Microsoft C++ Build Tools first
- **Linux**: `sudo apt-get install espeak espeak-data libespeak1 libespeak-dev build-essential`
- **macOS**: Install Xcode command line tools: `xcode-select --install`

**Session Issues:**
- Use "Clear All Work" button to reset corrupted sessions
- Check network connectivity for API calls
- Monitor API usage limits

**Performance:**
- Process large files one at a time
- Use text chunking for very long documents
- Monitor memory usage with multiple files

## 📊 API Usage & Limits

### OpenAI Integration
- **Daily Limit**: 20,000 API requests
- **Rate Limiting**: 20,000 tokens per minute
- **Usage Tracking**: Persistent across sessions
- **Smart Estimation**: ~4 characters per token

### Features Requiring API Key
- Grammar completion suggestions
- AI text improvements
- Text summarization
- OpenAI TTS (premium voices)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📜 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Streamlit** for the amazing web framework
- **OpenAI** for GPT-3.5 Turbo and TTS capabilities
- **LangGraph** for sophisticated agent orchestration
- **Tesseract & EasyOCR** for optical character recognition
- **Google TTS** for free text-to-speech functionality

---

Built with ❤️ by the Robin Reader team using cutting-edge AI and multi-agent systems.