# -*- coding: utf-8 -*-
import streamlit as st
import sys
import os
import traceback
from fact_check_bot import fact_check_url, fact_check_text

# Set page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Fact Check Agent",
    page_icon="🔍",
    layout="wide"
)

# Add the current directory to path to find the fact_check_bot module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up debug info
debug_info = []
error_details = None

# Import your fact-checking functionality
try:
    from fact_check_bot import fact_check_url, fact_check_text
    has_fact_check = True
except Exception as e:
    has_fact_check = False
    error_details = traceback.format_exc()
    debug_info.append(f"Import failed: {str(e)}")

# Title and description
st.title("🔍 Fact Check Agent")
st.write("Enter a URL or paste text to fact-check. The bot will analyze the content and provide a credibility assessment.")

# Create tabs for URL and text input
url_tab, text_tab = st.tabs(["Check URL", "Check Text"])

with url_tab:
    url = st.text_input("URL:", key="url_input")
    if st.button("Check Credibility", key="url_button"):
        if url:
            with st.spinner("Analyzing URL..."):
                result = fact_check_url(url)
                st.write(result)
        else:
            st.error("Please enter a URL")

with text_tab:
    text = st.text_area("Text to fact-check:", key="text_input")
    if st.button("Check Credibility", key="text_button"):
        if text:
            with st.spinner("Analyzing text..."):
                # Since we already have the content, we can skip the extraction step
                # by directly creating a modified crew with just the relevant tasks
                from fact_check_bot import (
                    claim_identifier, fact_researcher, claim_verifier, credibility_summarizer,
                    identify_claims_task, research_claims_task, verify_claims_task, summarize_credibility_task,
                    Crew, Process
                )
                
                # Create a text-specific crew that skips the content extraction task
                text_fact_check_crew = Crew(
                    agents=[
                        claim_identifier,
                        fact_researcher,
                        claim_verifier, 
                        credibility_summarizer
                    ],
                    tasks=[
                        identify_claims_task,
                        research_claims_task,
                        verify_claims_task,
                        summarize_credibility_task
                    ],
                    verbose=True,
                    process=Process.sequential
                )
                
                try:
                    # Pass the text directly to the crew
                    initial_context = {"cleaned_content": text}
                    result = text_fact_check_crew.kickoff(inputs=initial_context)
                    
                    # Get the final summary
                    if isinstance(result, dict) and "tasks_output" in result:
                        final_summary = result["tasks_output"][-1].raw
                        st.write(final_summary)
                    else:
                        st.write(str(result))
                        
                except Exception as e:
                    st.error(f"Error analyzing text: {str(e)}")
                    st.error(traceback.format_exc())
        else:
            st.error("Please enter some text")

# Add a sidebar with information
with st.sidebar:
    st.header("About")
    st.write("This fact-checking bot analyzes content to:")
    st.markdown("""
    * Identify factual claims
    * Research claims using reliable sources
    * Assess the credibility of information
    * Provide evidence-based verification
    """)
    
    st.header("Import Status") 
    if has_fact_check:
        st.success("Fact-checking module loaded successfully.")
    else:
        st.error("Using demo mode: Fact-checking module not available.")
        st.markdown("""
        To fix this:
        1. Check that fact_check_bot.py exists in the same directory
        2. Ensure all dependencies are installed 
        3. Check your .env file has the necessary API keys
        """)
        
        with st.expander("Debug Information"):
            st.markdown("### Debug Log")
            for log in debug_info:
                st.text(log)
            
            if error_details:
                st.markdown("### Error Details")
                st.code(error_details, language="python")
    
    st.markdown("---")
    
    st.header("How to run")
    st.code("""
    # Activate virtual environment
    # Windows:
    .\\venv\\Scripts\\activate
    
    # macOS/Linux:
    source venv/bin/activate
    
    # Run Streamlit app
    streamlit run app.py
    """)