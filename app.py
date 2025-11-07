import streamlit as st
import google.generativeai as genai
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Search Agent",
    page_icon="🤖",
    layout="centered",
)

# --- API Key and Model Configuration ---

# Try to get the API key from Streamlit's secrets
api_key = st.secrets.get("GEMINI_API_KEY")

# If the key is not found (e.g., running locally without setting secrets),
# fall back to an environment variable.
if not api_key:
    api_key = os.environ.get("GEMINI_API_KEY")

# If still not found, show an error and stop
if not api_key:
    st.error(
        "GEMINI_API_KEY not found. "
        "Please set it in Streamlit's secrets or as an environment variable."
    )
    st.stop()

# Configure the Gemini client
try:
    genai.configure(api_key=api_key)
    # Use gemini-2.5-flash-preview-09-2025 as it supports grounding
    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash-preview-09-2025',
        system_instruction=(
            "You are a helpful and concise search-powered assistant. "
            "Answer the user's query based *only* on the provided search results. "
            "Cite your sources."
        )
    )
except Exception as e:
    st.error(f"Error configuring the Gemini client: {e}")
    st.stop()

# Define the Google Search tool
google_search_tool = genai.Tool(
    google_search=genai.GoogleSearch()
)

# --- Main Application UI ---
st.title("🤖 AI Search Agent")
st.write(
    "Ask a question, and the agent will use Google Search to find the "
    "most up-to-date information and provide a synthesized answer."
)

# Use a form for the search input and button
with st.form(key="search_form"):
    user_query = st.text_input(
        "What would you like to know?",
        placeholder="e.g., What's the weather in London?",
        key="search_input"
    )
    submit_button = st.form_submit_button(label="Ask Agent")

# --- Backend Logic ---
if submit_button and user_query:
    with st.spinner("Searching and thinking..."):
        try:
            # Make the API call with the tool
            response = model.generate_content(
                user_query,
                tools=[google_search_tool],
            )

            # Extract the text and sources
            answer = response.text

            sources = []
            if response.grounding_metadata and response.grounding_metadata.grounding_attributions:
                for attribution in response.grounding_metadata.grounding_attributions:
                    if attribution.web:
                        sources.append({
                            'uri': attribution.web.uri,
                            'title': attribution.web.title
                        })

            # --- Display Results ---
            st.subheader("Answer")
            st.markdown(answer)

            if sources:
                st.subheader("Sources")
                for i, source in enumerate(sources):
                    # Format as: [1] [Title](uri)
                    st.markdown(f"[{i+1}] [{source['title']}]({source['uri']})")

        except Exception as e:
            # Handle potential API errors gracefully
            st.error(f"An error occurred: {e}")
            st.error(
                "This is the complete and final `requirements.txt` file. It lists the two Python libraries your app needs."
            )

elif submit_button and not user_query:
    st.warning("Please enter a question.")