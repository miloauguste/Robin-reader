"""
Streamlit UI Components for Parallel Grammar Editing
Provides side-by-side editing interface with grammar analysis and completion suggestions
"""

import streamlit as st
from typing import Dict, Any, Optional
from agents import ParallelEditingWindowAgent, TextReaderState
import json
from datetime import datetime

class ParallelEditorUI:
    """Streamlit UI components for parallel editing with grammar analysis"""
    
    def __init__(self):
        self.parallel_editor = ParallelEditingWindowAgent()
    
    def render_parallel_editor(self, state: TextReaderState, chunk_id: int, api_key: str = None) -> Dict[str, Any]:
        """Render the main parallel editing interface"""
        
        # Initialize session state for this chunk if needed
        session_key = f"editing_session_{chunk_id}"
        if session_key not in st.session_state:
            result = self.parallel_editor.create_parallel_editing_session(state, chunk_id)
            if result["success"]:
                st.session_state[session_key] = result["editing_session"]
            else:
                st.error(f"Failed to create editing session: {result['error']}")
                return {"action": "error", "error": result["error"]}
        
        editing_session = st.session_state[session_key]
        
        # Check for pending AI improvement request
        ai_improve_key = f"ai_improve_request_{chunk_id}"
        if ai_improve_key in st.session_state:
            improve_request = st.session_state[ai_improve_key]
            improvement_type = improve_request["improvement_type"]
            
            # Use the API key passed from the main app
            
            if api_key:
                with st.spinner(f"🤖 Improving text using OpenAI ({improvement_type.title()} style)..."):
                    try:
                        # Improve the text
                        result = self.parallel_editor.improve_text_with_openai(
                            editing_session, 
                            api_key, 
                            improvement_type
                        )
                        
                        if result["success"]:
                            # Update the session state with the improved text
                            st.session_state[session_key] = result["updated_session"]
                            editing_session = result["updated_session"]  # Update local variable
                            st.success(f"✅ Text improved using {improvement_type.title()} style!")
                        else:
                            st.error(f"❌ Text improvement failed: {result['error']}")
                    
                    except Exception as e:
                        st.error(f"❌ Text improvement error: {str(e)}")
            else:
                st.error("❌ OpenAI API key required for text improvement")
            
            # Clear the request
            del st.session_state[ai_improve_key]
        
        # Check for pending speech generation request
        speech_request_key = f"speech_request_{chunk_id}"
        if speech_request_key in st.session_state:
            speech_request = st.session_state[speech_request_key]
            
            with st.spinner("🎤 Generating speech from edited text..."):
                try:
                    # Get the text to convert to speech
                    text_to_speak = editing_session["parallel_text"]
                    
                    if not text_to_speak.strip():
                        st.error("❌ No text to convert to speech")
                    else:
                        # Import the TTS agent from the main session state
                        if hasattr(st.session_state, 'agents') and 'tts_agent' in st.session_state.agents:
                            tts_agent = st.session_state.agents['tts_agent']
                            
                            # Create a temporary state-like object for the TTS agent
                            temp_state = {
                                "processed_chunks": [{"current_text": text_to_speak}],
                                "tts_engine": getattr(st.session_state, 'current_tts_engine', "Google TTS (Free)"),
                                "tts_settings": getattr(st.session_state, 'current_tts_settings', {}),
                                "api_key": api_key
                            }
                            
                            # Generate speech for the single chunk
                            result_state = tts_agent.convert_chunk(temp_state, 0)
                            
                            if result_state.get("error_message"):
                                st.error(f"❌ Speech generation failed: {result_state['error_message']}")
                            else:
                                # Extract the audio data
                                audio_files = result_state.get("audio_files", [])
                                if audio_files:
                                    audio_data = audio_files[0]["audio_data"]
                                    st.success("🎉 Speech generated successfully!")
                                    st.audio(audio_data, format="audio/mp3")
                                    st.download_button(
                                        "💾 Download Audio",
                                        data=audio_data,
                                        file_name=f"parallel_edited_speech_{chunk_id}.mp3",
                                        mime="audio/mp3",
                                        key=f"download_parallel_speech_{chunk_id}"
                                    )
                                else:
                                    st.error("❌ No audio data generated")
                        else:
                            st.error("❌ TTS agent not available")
                
                except Exception as e:
                    st.error(f"❌ Speech generation error: {str(e)}")
            
            # Clear the request
            del st.session_state[speech_request_key]
        
        # Check for pending summarization request
        summarize_request_key = f"summarize_request_{chunk_id}"
        if summarize_request_key in st.session_state:
            summarize_request = st.session_state[summarize_request_key]
            
            with st.spinner("📋 Generating detailed summary with line references..."):
                try:
                    # Get the text to summarize
                    text_to_summarize = editing_session["parallel_text"]
                    
                    if not text_to_summarize.strip():
                        st.error("❌ No text to summarize")
                    elif not api_key:
                        st.error("❌ OpenAI API key required for text summarization")
                    else:
                        # Use the parallel editor agent to summarize
                        result = self.parallel_editor.summarize_text_with_openai(
                            editing_session, 
                            api_key, 
                            target_length=250
                        )
                        
                        if result["success"]:
                            # Update the session state with the summary
                            session_key = f"editing_session_{chunk_id}"
                            st.session_state[session_key] = result["updated_session"]
                            editing_session = result["updated_session"]  # Update local variable
                            
                            st.success("📋 Summary generated successfully!")
                            
                            # Display the summary in an expandable section
                            with st.expander("📋 Text Summary with Line References", expanded=True):
                                summary_text = result["summary_text"]
                                st.markdown("### Generated Summary")
                                st.markdown(summary_text)
                                
                                # Add copy button for the summary
                                st.download_button(
                                    "💾 Download Summary",
                                    data=summary_text,
                                    file_name=f"text_summary_chunk_{chunk_id}.txt",
                                    mime="text/plain",
                                    key=f"download_summary_{chunk_id}"
                                )
                                
                                # Show summary statistics
                                word_count = len(summary_text.split())
                                st.caption(f"Summary length: {word_count} words | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        else:
                            st.error(f"❌ Text summarization failed: {result['error']}")
                
                except Exception as e:
                    st.error(f"❌ Summarization error: {str(e)}")
            
            # Clear the request
            del st.session_state[summarize_request_key]
        
        # Check for pending export request
        export_request_key = f"export_request_{chunk_id}"
        if export_request_key in st.session_state:
            export_request = st.session_state[export_request_key]
            
            export_type = export_request["export_type"]
            include_speech = export_request["include_speech"]
            include_summary = export_request["include_summary"]
            summary_length = export_request["summary_length"]
            
            progress_text = []
            if include_speech:
                progress_text.append("🎤 Speech")
            if include_summary:
                progress_text.append("📋 Summary")
            progress_text.append("📄 HTML Export")
            
            with st.spinner(f"🚀 Generating {' + '.join(progress_text)}..."):
                try:
                    # Get the text to process
                    text_to_process = editing_session["parallel_text"]
                    
                    if not text_to_process.strip():
                        st.error("❌ No text to process")
                    elif (include_summary or (include_speech and st.session_state.current_tts_engine == "OpenAI TTS (Premium)")) and not api_key:
                        st.error("❌ OpenAI API key required for the selected options")
                    else:
                        # Get TTS settings from session state
                        tts_engine = getattr(st.session_state, 'current_tts_engine', "Google TTS (Free)")
                        tts_settings = getattr(st.session_state, 'current_tts_settings', {})
                        
                        # Use the parallel editor agent to generate combined export
                        result = self.parallel_editor.generate_speech_and_summary(
                            editing_session, 
                            api_key,
                            tts_engine=tts_engine,
                            tts_settings=tts_settings,
                            include_speech=include_speech,
                            include_summary=include_summary,
                            summary_length=summary_length
                        )
                        
                        if result["success"]:
                            # Update the session state
                            session_key = f"editing_session_{chunk_id}"
                            st.session_state[session_key] = result["updated_session"]
                            editing_session = result["updated_session"]  # Update local variable
                            
                            export_data = result["export_data"]
                            
                            st.success(f"🚀 {export_type.title().replace('_', ' ')} export completed!")
                            
                            # Display results in an expandable section
                            with st.expander("🚀 Export Results", expanded=True):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("### 📊 Export Summary")
                                    st.info(f"**Type:** {export_type.title().replace('_', ' ')}")
                                    st.info(f"**Text:** {len(text_to_process)} chars, {len(text_to_process.split())} words")
                                    
                                    if export_data["audio_data"]:
                                        audio_info = export_data["audio_info"]
                                        st.info(f"**Speech:** {audio_info['engine']}")
                                        duration_est = audio_info['word_count'] / 150
                                        st.info(f"**Duration:** ~{duration_est:.1f} minutes")
                                    
                                    if export_data["summary_text"]:
                                        summary_words = len(export_data["summary_text"].split())
                                        st.info(f"**Summary:** {summary_words} words")
                                
                                with col2:
                                    st.markdown("### 💾 Download Options")
                                    
                                    # HTML download button (always available)
                                    html_content = export_data["html_content"]
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    html_filename = f"text_export_chunk_{chunk_id}_{timestamp}.html"
                                    
                                    st.download_button(
                                        "📄 Download HTML Export",
                                        data=html_content,
                                        file_name=html_filename,
                                        mime="text/html",
                                        key=f"download_html_{chunk_id}",
                                        help="Complete HTML file with embedded audio and content"
                                    )
                                    
                                    # Individual download options
                                    if export_data["audio_data"]:
                                        st.download_button(
                                            "🎵 Download Audio Only",
                                            data=export_data["audio_data"],
                                            file_name=f"speech_chunk_{chunk_id}_{timestamp}.mp3",
                                            mime="audio/mp3",
                                            key=f"download_audio_only_{chunk_id}"
                                        )
                                    
                                    if export_data["summary_text"]:
                                        st.download_button(
                                            "📋 Download Summary Only",
                                            data=export_data["summary_text"],
                                            file_name=f"summary_chunk_{chunk_id}_{timestamp}.txt",
                                            mime="text/plain",
                                            key=f"download_summary_only_{chunk_id}"
                                        )
                                
                                # Preview sections
                                if export_data["audio_data"]:
                                    st.markdown("### 🎵 Audio Preview")
                                    st.audio(export_data["audio_data"], format="audio/mp3")
                                
                                if export_data["summary_text"]:
                                    st.markdown("### 📋 Summary Preview")
                                    st.markdown(export_data["summary_text"])
                        else:
                            st.error(f"❌ Export failed: {result['error']}")
                
                except Exception as e:
                    st.error(f"❌ Export error: {str(e)}")
            
            # Clear the request
            del st.session_state[export_request_key]
        
        st.subheader(f"📝 Parallel Editor - Chunk {chunk_id + 1}")
        
        # Display grammar analysis summary
        if editing_session.get("needs_completion", False):
            grammar_analysis = editing_session.get("grammar_analysis", {})
            incomplete_count = len(grammar_analysis.get("incomplete_sentences", []))
            total_count = grammar_analysis.get("total_sentences", 0)
            completeness_score = grammar_analysis.get("completeness_score", 0) * 100
            
            st.warning(f"⚠️ Found {incomplete_count} incomplete sentences out of {total_count} total sentences. "
                      f"Completeness score: {completeness_score:.1f}%")
        else:
            st.success("✅ All sentences appear to be grammatically complete!")
        
        # Create two columns for parallel editing
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📄 Original Text")
            st.text_area(
                "Original",
                value=editing_session["original_text"],
                height=300,
                disabled=True,
                key=f"original_{chunk_id}"
            )
            
            # Display completion suggestions
            completion_suggestions = editing_session.get("completion_suggestions", [])
            if completion_suggestions:
                st.markdown("### 💡 Completion Suggestions")
                for i, suggestion in enumerate(completion_suggestions):
                    with st.expander(f"Suggestion {i + 1}: {suggestion['issues']}"):
                        st.markdown(f"**Original:** {suggestion['original']}")
                        st.markdown(f"**Suggested:** {suggestion['completed']}")
                        
                        if st.button(f"Apply Suggestion {i + 1}", key=f"apply_{chunk_id}_{i}"):
                            result = self.parallel_editor.apply_completion_suggestion(editing_session, i)
                            if result["success"]:
                                st.session_state[session_key] = result["updated_session"]
                                st.rerun()
                            else:
                                st.error(f"Failed to apply suggestion: {result['error']}")
        
        with col2:
            st.markdown("### ✏️ Editable Text")
            
            # OpenAI Text Improvement Controls
            col2_1, col2_2 = st.columns([2, 1])
            
            with col2_1:
                improvement_type = st.selectbox(
                    "AI Improvement Type",
                    options=["general", "professional", "casual", "concise", "detailed", "formal"],
                    format_func=lambda x: x.title(),
                    key=f"improvement_type_{chunk_id}",
                    help="Choose the type of text improvement to apply"
                )
            
            with col2_2:
                improve_clicked = st.button("🤖 Improve with AI", key=f"improve_{chunk_id}", help="Use OpenAI to improve the text")
                
                # Handle AI improvement within the UI component
                if improve_clicked:
                    # Store the improvement request in session state to be processed
                    st.session_state[f"ai_improve_request_{chunk_id}"] = {
                        "improvement_type": improvement_type,
                        "chunk_id": chunk_id
                    }
                    st.rerun()
            
            # Editable text area with dynamic key for proper refresh
            # Add timestamp to key to force refresh after AI improvements
            refresh_key = editing_session.get("last_ai_improvement", {}).get("applied_at", "")
            text_area_key = f"parallel_{chunk_id}_{hash(refresh_key) if refresh_key else ''}"
            
            edited_text = st.text_area(
                "Edit your text here",
                value=editing_session["parallel_text"],
                height=300,
                key=text_area_key,
                help="Make your edits here or use AI improvement above. Changes will be highlighted when you approve."
            )
            
            # Update parallel text if changed
            if edited_text != editing_session["parallel_text"]:
                result = self.parallel_editor.update_parallel_text(editing_session, edited_text)
                if result["success"]:
                    session_key = f"editing_session_{chunk_id}"
                    st.session_state[session_key] = result["updated_session"]
            
            # Show change indicator and AI improvement status
            if editing_session.get("last_summary"):
                summary_info = editing_session["last_summary"]
                word_count = len(summary_info["summary_text"].split())
                st.info(f"📋 Summary generated ({word_count} words)")
            
            if editing_session.get("last_ai_improvement"):
                improvement_info = editing_session["last_ai_improvement"]
                st.success(f"🤖 AI improved ({improvement_info['improvement_type'].title()}) - Changes detected")
            elif editing_session.get("changes_made", False):
                st.info("📝 Changes detected")
            else:
                st.success("💾 No changes")
        
        # Combined export options
        st.markdown("---")
        st.subheader("🚀 Export Options")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            st.markdown("**Individual Options:**")
            individual_col1, individual_col2, individual_col3 = st.columns(3)
            
            with individual_col1:
                generate_speech_clicked = st.button("🎤 Speech Only", key=f"speech_only_{chunk_id}", help="Generate speech audio only")
            
            with individual_col2:
                generate_summary_clicked = st.button("📋 Summary Only", key=f"summary_only_{chunk_id}", help="Generate text summary only")
            
            with individual_col3:
                generate_both_clicked = st.button("🎵📋 Both + HTML", key=f"both_html_{chunk_id}", help="Generate speech, summary, and HTML export")
        
        with export_col2:
            st.markdown("**Export Configuration:**")
            
            config_col1, config_col2 = st.columns(2)
            with config_col1:
                include_speech = st.checkbox("Include Speech", value=True, key=f"include_speech_{chunk_id}")
                include_summary = st.checkbox("Include Summary", value=True, key=f"include_summary_{chunk_id}")
            
            with config_col2:
                summary_length = st.number_input(
                    "Summary Length (words)", 
                    min_value=100, 
                    max_value=500, 
                    value=250, 
                    step=50,
                    key=f"summary_length_{chunk_id}"
                )
            
            custom_export_clicked = st.button("🎯 Custom Export", key=f"custom_export_{chunk_id}", help="Generate custom combination and HTML export")
        
        # Handle export actions
        export_action = None
        
        if generate_speech_clicked:
            export_action = {"type": "speech_only", "include_speech": True, "include_summary": False}
        elif generate_summary_clicked:
            export_action = {"type": "summary_only", "include_speech": False, "include_summary": True}
        elif generate_both_clicked:
            export_action = {"type": "both_html", "include_speech": True, "include_summary": True}
        elif custom_export_clicked:
            if not include_speech and not include_summary:
                st.error("❌ Please select at least one option (Speech or Summary)")
            else:
                export_action = {"type": "custom", "include_speech": include_speech, "include_summary": include_summary}
        
        if export_action:
            # Store the export request in session state
            st.session_state[f"export_request_{chunk_id}"] = {
                "chunk_id": chunk_id,
                "editing_session": editing_session,
                "export_type": export_action["type"],
                "include_speech": export_action["include_speech"],
                "include_summary": export_action["include_summary"],
                "summary_length": summary_length
            }
            st.rerun()
        
        # Action buttons
        st.markdown("---")
        st.subheader("📝 Text Actions")
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button("✅ Approve Changes", key=f"approve_{chunk_id}"):
                if editing_session.get("changes_made", False):
                    return {"action": "approve", "editing_session": editing_session}
                else:
                    st.warning("No changes to approve")
        
        with action_col2:
            if st.button("❌ Reject Changes", key=f"reject_{chunk_id}"):
                result = self.parallel_editor.reject_parallel_edits(editing_session)
                if result["success"]:
                    session_key = f"editing_session_{chunk_id}"
                    st.session_state[session_key] = result["updated_session"]
                    st.rerun()
        
        with action_col3:
            if st.button("🔍 Show Comparison", key=f"compare_{chunk_id}"):
                return {"action": "show_comparison", "editing_session": editing_session}
        
        return {"action": "continue", "editing_session": editing_session}
    
    def render_comparison_view(self, editing_session: Dict[str, Any]):
        """Render side-by-side comparison of original vs edited text"""
        
        comparison = self.parallel_editor.get_editing_comparison(editing_session)
        
        if "error" in comparison:
            st.error(f"Error generating comparison: {comparison['error']}")
            return
        
        st.subheader("🔍 Text Comparison")
        
        # Statistics
        stats = comparison["statistics"]
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Original Characters", stats["original_chars"])
        with col2:
            st.metric("Edited Characters", stats["parallel_chars"])
        with col3:
            st.metric("Character Difference", 
                     f"{stats['char_difference']:+d}",
                     delta=stats['char_difference'])
        
        # Side-by-side text comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📄 Original")
            st.text_area("Original Text", comparison["original_text"], height=400, disabled=True)
        
        with col2:
            st.markdown("### ✏️ Edited")
            st.text_area("Edited Text", comparison["parallel_text"], height=400, disabled=True)
        
        # Diff view
        if comparison["diff"]:
            st.markdown("### 🔄 Differences (Unified Diff)")
            diff_text = "\n".join(comparison["diff"])
            st.code(diff_text, language="diff")
        
        # Grammar analysis summary
        grammar_analysis = comparison.get("grammar_analysis", {})
        if grammar_analysis:
            st.markdown("### 📊 Grammar Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Sentences", grammar_analysis.get("total_sentences", 0))
                st.metric("Complete Sentences", len(grammar_analysis.get("complete_sentences", [])))
            
            with col2:
                st.metric("Incomplete Sentences", len(grammar_analysis.get("incomplete_sentences", [])))
                completeness = grammar_analysis.get("completeness_score", 0) * 100
                st.metric("Completeness Score", f"{completeness:.1f}%")
    
    def render_grammar_analysis_details(self, editing_session: Dict[str, Any]):
        """Render detailed grammar analysis information"""
        
        grammar_analysis = editing_session.get("grammar_analysis", {})
        if not grammar_analysis:
            st.info("No grammar analysis available")
            return
        
        st.subheader("📝 Grammar Analysis Details")
        
        # Complete sentences
        complete_sentences = grammar_analysis.get("complete_sentences", [])
        if complete_sentences:
            st.markdown("### ✅ Complete Sentences")
            for sentence in complete_sentences:
                st.success(f"**Sentence {sentence['index'] + 1}:** {sentence['text']}")
        
        # Incomplete sentences with details
        incomplete_sentences = grammar_analysis.get("incomplete_sentences", [])
        if incomplete_sentences:
            st.markdown("### ⚠️ Incomplete Sentences")
            for sentence in incomplete_sentences:
                with st.expander(f"Issue: {', '.join(sentence['issues'])}"):
                    st.write(f"**Text:** {sentence['text']}")
                    st.write(f"**Issues:** {sentence['issues']}")
                    
                    issue_details = []
                    if not sentence['has_subject']:
                        issue_details.append("❌ Missing subject")
                    else:
                        issue_details.append("✅ Has subject")
                    
                    if not sentence['has_predicate']:
                        issue_details.append("❌ Missing predicate")
                    else:
                        issue_details.append("✅ Has predicate")
                    
                    if sentence['is_fragment']:
                        issue_details.append("❌ Sentence fragment")
                    
                    if sentence['is_run_on']:
                        issue_details.append("❌ Run-on sentence")
                    
                    st.markdown("**Analysis:**")
                    for detail in issue_details:
                        st.markdown(f"- {detail}")
        
        # Overall statistics
        st.markdown("### 📊 Summary Statistics")
        total = grammar_analysis.get("total_sentences", 0)
        complete = len(complete_sentences)
        incomplete = len(incomplete_sentences)
        score = grammar_analysis.get("completeness_score", 0) * 100
        
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        with metrics_col1:
            st.metric("Total Sentences", total)
        with metrics_col2:
            st.metric("Complete", complete)
        with metrics_col3:
            st.metric("Incomplete", incomplete)
        with metrics_col4:
            st.metric("Completeness Score", f"{score:.1f}%")
    
    def render_speech_preview(self, editing_session: Dict[str, Any]):
        """Render speech preview options"""
        
        st.subheader("🎤 Speech Generation Preview")
        
        text_to_speak = editing_session["parallel_text"]
        
        st.markdown("### Text to be converted to speech:")
        st.text_area("Speech Text", text_to_speak, height=150, disabled=True)
        
        # Speech options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔧 Speech Options")
            voice_speed = st.slider("Speech Speed", 0.5, 2.0, 1.0, 0.1)
            voice_option = st.selectbox(
                "Voice Type",
                ["Google TTS", "OpenAI TTS (Premium)"],
                help="Choose between different TTS engines"
            )
        
        with col2:
            st.markdown("### 📊 Text Statistics")
            word_count = len(text_to_speak.split())
            char_count = len(text_to_speak)
            estimated_duration = word_count / 150  # ~150 words per minute average
            
            st.metric("Word Count", word_count)
            st.metric("Character Count", char_count)
            st.metric("Estimated Duration", f"{estimated_duration:.1f} minutes")
        
        if st.button("🎵 Generate Speech"):
            return {
                "action": "generate_speech",
                "text": text_to_speak,
                "voice_speed": voice_speed,
                "voice_option": voice_option
            }
        
        return {"action": "preview_only"}
    
    def render_speech_generation_result(self, audio_result: Dict[str, Any]):
        """Render the results of speech generation"""
        
        if audio_result["success"]:
            st.success("🎉 Speech generated successfully!")
            
            audio_info = audio_result["audio_info"]
            
            # Display audio information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Text Length", f"{audio_info['text_length']} chars")
            with col2:
                st.metric("Word Count", audio_info["word_count"])
            with col3:
                st.metric("Engine", audio_info["engine"])
            
            # Audio player
            st.markdown("### 🔊 Generated Audio")
            
            # Create audio player using streamlit audio component
            audio_bytes = audio_result["audio_data"]
            if isinstance(audio_bytes, bytes):
                st.audio(audio_bytes, format="audio/mp3")
            else:
                st.error("Invalid audio data format")
            
            # Download button
            st.download_button(
                label="💾 Download Audio",
                data=audio_bytes,
                file_name=f"speech_{audio_info['session_id']}.mp3",
                mime="audio/mp3"
            )
            
        else:
            st.error(f"❌ Speech generation failed: {audio_result['error']}")
    
    def render_full_workflow_interface(self, state: TextReaderState, chunk_id: int, api_key: str = None):
        """Render the complete workflow interface for grammar checking and editing"""
        
        st.markdown("---")
        st.subheader("🔄 Grammar Checking & Editing Workflow")
        
        # Step indicators
        steps_col1, steps_col2, steps_col3, steps_col4 = st.columns(4)
        
        with steps_col1:
            st.markdown("**1️⃣ Analyze**")
            st.caption("Grammar check")
        
        with steps_col2:
            st.markdown("**2️⃣ Edit**")
            st.caption("Parallel editing")
        
        with steps_col3:
            st.markdown("**3️⃣ Approve**")
            st.caption("Review changes")
        
        with steps_col4:
            st.markdown("**4️⃣ Speak**")
            st.caption("Generate audio")
        
        # Main workflow
        workflow_action = self.render_parallel_editor(state, chunk_id, api_key)
        
        if workflow_action["action"] == "approve":
            # Handle approval
            st.success("✅ Changes approved! Text has been updated.")
            
            # Show option to proceed to speech generation
            if st.button("🎤 Proceed to Speech Generation", key=f"proceed_speech_{chunk_id}"):
                st.session_state[f"show_speech_{chunk_id}"] = True
                st.rerun()
        
        elif workflow_action["action"] == "show_comparison":
            # Show comparison view
            st.markdown("---")
            self.render_comparison_view(workflow_action["editing_session"])
        
        elif workflow_action["action"] == "generate_speech":
            # Handle speech generation
            st.markdown("---")
            speech_result = self.render_speech_preview(workflow_action["editing_session"])
            
            if speech_result["action"] == "generate_speech":
                # This would integrate with the actual TTS generation
                st.info("🔄 Speech generation would be handled by the backend agents...")
                
                # Mock result for demonstration
                mock_audio_result = {
                    "success": True,
                    "audio_data": b"mock_audio_data",  # In reality, this would be actual audio bytes
                    "audio_info": {
                        "text_length": len(speech_result["text"]),
                        "word_count": len(speech_result["text"].split()),
                        "engine": speech_result["voice_option"],
                        "session_id": workflow_action["editing_session"]["session_id"]
                    }
                }
                
                # Note: In the actual implementation, you would call:
                # from agents import HumanEditingAgent
                # human_editor = HumanEditingAgent()
                # audio_result = human_editor.generate_speech_from_parallel_edit(
                #     state, workflow_action["editing_session"], 
                #     speech_result["voice_option"], 
                #     {"speed": speech_result["voice_speed"]}
                # )
                # self.render_speech_generation_result(audio_result)
        
        # Show speech generation interface if requested
        if st.session_state.get(f"show_speech_{chunk_id}", False):
            st.markdown("---")
            speech_action = self.render_speech_preview(workflow_action["editing_session"])
            
            if speech_action["action"] == "generate_speech":
                # Actual speech generation would happen here
                st.success("🎵 Speech generation completed!")
                st.session_state[f"show_speech_{chunk_id}"] = False

def main():
    """Demo function for testing the parallel editor UI"""
    st.title("Parallel Grammar Editor Demo")
    
    # This would normally be called from the main app with real state
    st.info("This is a demo of the Parallel Grammar Editor UI components. "
           "In the actual app, this would be integrated with the text processing workflow.")
    
    editor_ui = ParallelEditorUI()
    
    # Mock data for demonstration
    mock_text = """This is complete sentence. Because it has both. Running very long sentence that goes on and on without proper punctuation and structure which makes it difficult to read and understand what the author is trying to say. The weather nice today."""
    
    st.markdown("### Sample Text with Grammar Issues")
    st.text_area("Sample", mock_text, height=100, disabled=True)
    
    st.markdown("**Issues in the sample text:**")
    st.markdown("- 'Because it has both.' - Sentence fragment")
    st.markdown("- Long run-on sentence")
    st.markdown("- 'The weather nice today.' - Missing predicate")

if __name__ == "__main__":
    main()