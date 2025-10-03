"""
Grammar Checker Integration Demo
Demonstrates how to integrate the grammar checking and parallel editing features
into the existing text reader application.
"""

import streamlit as st
from agents import (
    GrammarAnalysisAgent, 
    ParallelEditingWindowAgent, 
    HumanEditingAgent,
    TextReaderState
)
from parallel_editor_ui import ParallelEditorUI
from typing import Dict, Any, List

def initialize_demo_state() -> TextReaderState:
    """Initialize a demo state with sample text chunks"""
    
    sample_texts = [
        """This is complete sentence with proper grammar. However, this one missing verb. Because it starts with subordinating conjunction. The weather nice today and everyone enjoying the sunshine which makes for a perfect day to go outside and play sports or have picnic with family and friends and maybe even go swimming if there is pool nearby.""",
        
        """Machine learning revolutionizing many industries. The algorithms can process data. Very quickly and accurately. Companies using AI to improve customer experience, reduce costs, and increase efficiency in their operations, which leads to better profit margins and competitive advantages in the marketplace.""",
        
        """Climate change affecting global weather patterns. Scientists studying these changes. To understand impact. The temperature rising every year and causing more extreme weather events like hurricanes, floods, droughts, and heat waves that affect millions of people around the world."""
    ]
    
    processed_chunks = []
    for i, text in enumerate(sample_texts):
        chunk_data = {
            "id": i,
            "original_text": text,
            "current_text": text,
            "spell_checked": False,
            "grammar_checked": False,
            "human_edited": False,
            "char_count": len(text)
        }
        processed_chunks.append(chunk_data)
    
    return {
        "processed_chunks": processed_chunks,
        "current_step": "text_processing",
        "api_key": None,  # Would be set by user in actual app
        "editing_mode": True
    }

def main():
    """Main demo application"""
    
    st.set_page_config(
        page_title="Grammar Checker Demo",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("📝 Grammar Checker & Parallel Editor Demo")
    st.markdown("*Analyze sentence completeness, edit text in parallel window, and generate speech*")
    
    # Initialize demo state
    if "demo_state" not in st.session_state:
        st.session_state.demo_state = initialize_demo_state()
    
    state = st.session_state.demo_state
    
    # API Key input (for LLM-based completion)
    with st.sidebar:
        st.header("🔧 Configuration")
        api_key = st.text_input("OpenAI API Key (for LLM completion)", type="password")
        if api_key:
            state["api_key"] = api_key
        
        st.markdown("---")
        st.header("📊 Demo Info")
        st.info("This demo shows the grammar checking and parallel editing features. "
               "Upload your own OpenAI API key to enable LLM-based sentence completion.")
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["📝 Parallel Editor", "📊 Grammar Analysis", "🔍 Feature Overview"])
    
    with tab1:
        st.header("Parallel Editing Interface")
        
        # Chunk selection
        chunks = state.get("processed_chunks", [])
        if chunks:
            chunk_options = [f"Chunk {i+1} ({len(chunk['current_text'])} chars)" 
                           for i, chunk in enumerate(chunks)]
            
            selected_chunk_idx = st.selectbox(
                "Select text chunk to edit:",
                range(len(chunk_options)),
                format_func=lambda x: chunk_options[x]
            )
            
            # Show the parallel editor for the selected chunk
            editor_ui = ParallelEditorUI()
            editor_ui.render_full_workflow_interface(state, selected_chunk_idx)
        else:
            st.warning("No text chunks available for editing")
    
    with tab2:
        st.header("Grammar Analysis Details")
        
        if chunks:
            analysis_chunk_idx = st.selectbox(
                "Select chunk for analysis:",
                range(len(chunks)),
                format_func=lambda x: f"Chunk {x+1}",
                key="analysis_selection"
            )
            
            # Perform grammar analysis
            grammar_agent = GrammarAnalysisAgent()
            chunk = chunks[analysis_chunk_idx]
            text = chunk["current_text"]
            
            # Analyze the text
            analysis = grammar_agent.analyze_sentence_completeness(text)
            
            # Display results
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
            
            # Show detailed analysis
            st.subheader("📄 Original Text")
            st.text_area("Text being analyzed", text, height=150, disabled=True)
            
            # Complete sentences
            if analysis["complete_sentences"]:
                st.subheader("✅ Complete Sentences")
                for sentence in analysis["complete_sentences"]:
                    st.success(f"**{sentence['index'] + 1}.** {sentence['text']}")
            
            # Incomplete sentences
            if analysis["incomplete_sentences"]:
                st.subheader("⚠️ Incomplete Sentences")
                for sentence in analysis["incomplete_sentences"]:
                    issues = ", ".join(sentence["issues"])
                    st.error(f"**{sentence['index'] + 1}.** {sentence['text']}")
                    st.caption(f"Issues: {issues}")
                    
                    # Show LLM completion if API key is provided
                    if api_key:
                        with st.expander(f"💡 AI Completion Suggestion for Sentence {sentence['index'] + 1}"):
                            try:
                                completed = grammar_agent.complete_sentence_with_llm(
                                    sentence["text"], 
                                    text[:200],  # Context
                                    api_key
                                )
                                st.markdown(f"**Original:** {sentence['text']}")
                                st.markdown(f"**Completed:** {completed}")
                            except Exception as e:
                                st.error(f"Error generating completion: {str(e)}")
            
        else:
            st.warning("No text chunks available for analysis")
    
    with tab3:
        st.header("🚀 Feature Overview")
        
        st.markdown("""
        ### Grammar Checking & Parallel Editing Features
        
        This implementation provides comprehensive grammar checking with parallel editing capabilities:
        
        #### 🔍 **Grammar Analysis**
        - **Sentence Completeness Detection**: Identifies missing subjects, predicates, and sentence fragments
        - **Run-on Sentence Detection**: Flags overly long sentences that may need restructuring
        - **Grammar Issue Classification**: Categorizes different types of grammatical problems
        - **Completeness Scoring**: Provides an overall assessment of text quality
        
        #### ✏️ **Parallel Editing Interface**
        - **Side-by-Side View**: Original text on left, editable version on right
        - **Real-time Change Detection**: Highlights when modifications are made
        - **Grammar-Aware Editing**: Shows completion suggestions alongside the editor
        - **Diff Visualization**: Compare original vs edited text with unified diff view
        
        #### 🤖 **LLM-Powered Completion**
        - **Context-Aware Suggestions**: Uses surrounding text to generate appropriate completions
        - **Issue-Specific Fixes**: Targets specific grammar problems (fragments, missing subjects, etc.)
        - **Preserve Original Intent**: Maintains the author's intended meaning and style
        - **Rate-Limited API Usage**: Manages OpenAI API calls efficiently
        
        #### ✅ **Approval Workflow**
        - **Review Changes**: Examine all modifications before applying
        - **Selective Application**: Choose which suggestions to accept
        - **Revert Capability**: Undo changes if needed
        - **Change Tracking**: Maintain history of all modifications
        
        #### 🎤 **Speech Generation**
        - **Text-to-Speech Integration**: Convert edited text to natural-sounding audio
        - **Multiple Engines**: Support for Google TTS and OpenAI TTS
        - **Voice Customization**: Adjust speed, voice type, and other parameters
        - **Audio Download**: Save generated speech files
        
        #### 🔧 **Integration Features**
        - **Streamlit UI Components**: Ready-to-use interface components
        - **State Management**: Persistent editing sessions
        - **Error Handling**: Robust error recovery and user feedback
        - **Extensible Architecture**: Easy to add new grammar rules and features
        
        ### Usage Workflow
        
        1. **Upload Text**: Load documents through the existing file processing system
        2. **Analyze Grammar**: Run sentence completeness analysis on text chunks
        3. **Edit in Parallel**: Use the side-by-side editor to make improvements
        4. **Apply Suggestions**: Use AI-powered completion suggestions
        5. **Review Changes**: Compare original vs edited versions
        6. **Approve Edits**: Apply changes to the main text
        7. **Generate Speech**: Convert the improved text to audio
        
        ### Technical Implementation
        
        - **GrammarAnalysisAgent**: Core grammar checking logic
        - **ParallelEditingWindowAgent**: Parallel editing session management
        - **ParallelEditorUI**: Streamlit interface components
        - **HumanEditingAgent**: Workflow orchestration and speech integration
        - **Integration Points**: Works with existing LangGraph workflow
        """)
        
        # Show sample problematic text for demonstration
        st.subheader("📝 Sample Text with Grammar Issues")
        sample_text = """This is complete sentence. Because it has both. Running very long sentence that goes on and on without proper punctuation and structure which makes it difficult to read and understand what the author is trying to say. The weather nice today."""
        
        st.text_area("Sample Text", sample_text, height=100, disabled=True)
        
        st.markdown("**Issues in the sample text:**")
        st.markdown("- `'Because it has both.'` - Sentence fragment (starts with subordinating conjunction)")
        st.markdown("- Third sentence - Run-on sentence (too long, needs restructuring)")
        st.markdown("- `'The weather nice today.'` - Missing predicate (verb)")
        
        st.info("💡 Try selecting this sample text in the Parallel Editor tab to see the grammar analysis and editing features in action!")

if __name__ == "__main__":
    main()