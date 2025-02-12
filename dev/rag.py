import streamlit as st
from typing import Optional, List
import requests
from datetime import datetime, timezone
from urllib.parse import urlparse
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from supabase import create_client
from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables and initialize clients
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("ERROR: Supabase URL and SERVICE KEY must be set in .env")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
client = Groq()

# Configure Streamlit page
st.set_page_config(
    page_title="CPS AI Assistant",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "AI Assistant for Northeastern University's College of Professional Studies"
    }
)

# Apply custom styles
st.markdown("""
    <style>
    /* Main styles */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Title styling */
    .big-font {
        font-size: 30px !important;
        font-weight: bold;
        color: white;
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 16px;
        color: #FAFAFA;
        margin-bottom: 2rem;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: transparent;
        color: white;
        border: 1px solid white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    /* Button hover effect */
    .stButton>button:hover {
        background-color: rgba(255, 255, 255, 0.1);
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }

    /* Text color */
    .stMarkdown {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

class OllamaEmbeddings:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.endpoint = f"{base_url}/api/embeddings"
        
    def embed_query(self, text):
        response = requests.post(
            self.endpoint,
            json={
                "model": "nomic-embed-text:latest",
                "prompt": text
            }
        )
        response_data = response.json()
        if "embedding" not in response_data:
            raise ValueError(f"ERROR: Failed to get embedding. Response: {response_data}")
        return response_data["embedding"]
    
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

def get_available_programs():
    """Fetch available programs from Supabase"""
    try:
        response = supabase.from_('site_pages')\
            .select('title')\
            .execute()
        
        programs = set()
        for row in response.data:
            if row['title']:
                title = row['title'].strip()
                if title:
                    programs.add(title)
        
        return sorted(list(programs))
    except Exception as e:
        st.error(f"Error fetching programs: {str(e)}")
        return []

def generate_prompt(query: str, context: str) -> str:
    prompt_template = """ You are an AI assistant for the College of Professional Studies at Northeastern University, providing detailed and relevant information about course programs.
    If you cannot find the relevant information, ask the user to start a new search for course specific search results in this assistant.  
**Instructions:**  
- Use the provided context to answer the query clearly and concisely.  
- Format the response in markdown for readability.  
- Include useful URLs for further details.  
- If no relevant information is found from the context to user query, ask the user to Search in course specific section of this Assistant.
- If the context lacks sufficient details, state the limitation and suggest the user to search for course specific queries of this Assistant 

**Context:**  
{context}  

**User Query:**  
{query}  
    """
    
    return prompt_template.format(context=context, query=query)

def concatenate_chunks(chunks: List[dict], max_length: int = 100000) -> str:
    context_parts = []
    current_length = 0
    
    for chunk in chunks:
        chunk_text = f"[From {chunk.get('title', 'Unknown Source')}]\n{chunk.get('content', '')}\n\n"
        
        if current_length + len(chunk_text) <= max_length:
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        else:
            break
    
    return "".join(context_parts)

def stream_groq_response(prompt: str) -> None:
    try:
        chat = ChatGroq(
            model="llama3-70b-8192",
            streaming=True,
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        response_container = st.empty()
        full_response = ""
        
        for chunk in chat.stream(prompt):
            full_response += chunk.content
            response_container.markdown(full_response)
        
        return full_response
    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

def process_search_results(query: str, results: List[dict]) -> None:
    if not results:
        st.warning("No results found for your query.")
        return
    
    context = concatenate_chunks(results)
    rag_prompt = generate_prompt(query, context)
    
    #st.write("Generating response...")
    stream_groq_response(rag_prompt)

def initialize_session_state():
    if 'selected_program' not in st.session_state:
        st.session_state.selected_program = None
    if 'search_mode' not in st.session_state:
        st.session_state.search_mode = None
    if 'program_search' not in st.session_state:
        st.session_state.program_search = ""
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""

def reset_session():
    st.session_state.selected_program = None
    st.session_state.search_mode = None
    st.session_state.program_search = ""
    st.session_state.last_query = ""
    st.rerun()

def filter_programs(programs, search_term):
    if not search_term:
        return programs
    search_term = search_term.lower()
    return [prog for prog in programs if search_term in prog.lower()]

def get_relevant_chunks(query, program_name: Optional[str] = None, top_k=10):
    try:
        embeddings = OllamaEmbeddings()
        query_embedding = embeddings.embed_query(query)
        
        filter_params = {
            'source': 'cps_program_docs'
        }
        if program_name:
            filter_params['program_name'] = program_name
        
        response = supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': top_k,
                'search_mode': 'general' if program_name == "" else 'specific',
                'filter': filter_params
            }
        ).execute()
        
        return response.data or []
        
    except Exception as e:
        st.error(f"Error in search: {str(e)}")
        return []

def main():
    # Title section with custom styling
    st.markdown('<p class="big-font">🐾 AI Assistant for CPS Programs</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Hey Huskies! 👋 Get instant answers about CPS programs, courses, and requirements. '
        'Choose between a general search or dive deep into specific programs.</p>', 
        unsafe_allow_html=True
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Create a sidebar for the reset button when in search mode
    if st.session_state.search_mode is not None:
        with st.sidebar:
            if st.button("↩️ Start New Search", key="reset_button"):
                reset_session()
    
    # If search mode not selected, show initial options
    if st.session_state.search_mode is None:
        st.write("#### Choose Your Search Mode:")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Program-Specific Search"):
                st.session_state.search_mode = "specific"
                st.rerun()
        with col2:
            if st.button("🔍 General Search"):
                st.session_state.search_mode = "general"
                st.session_state.selected_program = ""
                st.rerun()
    
    # Program selection
    if st.session_state.search_mode == "specific":
        all_programs = get_available_programs()
        
        if all_programs:
            st.session_state.selected_program = st.selectbox(
                "📚 Select a Program:",
                options=all_programs,
                key="program_selector"
            )
        else:
            st.error("❌ No programs available.")
    
    # Search interface
    if st.session_state.search_mode is not None:
        st.write("---")
        
        # Current mode display
        mode_text = "🔍 General Search" if st.session_state.search_mode == "general" else "🎯 Program-Specific Search"
        st.markdown(f"**Current Mode:** {mode_text}")
        
        if st.session_state.search_mode == "specific" and st.session_state.selected_program:
            st.markdown(f"**Selected Program:** 📚 {st.session_state.selected_program}")

        placeholder = "Examples: What is the course structure? Or ask anything you'd like to know and hit enter!"
        if st.session_state.search_mode == "specific":
            placeholder = "Examples: What is the course structure? Or ask anything you'd like to know about the program and hit enter!"
        else:
            placeholder = "Examples: Compare between two courses! Or ask anything you'd like to know about any program and hit enter!"
        
        # Search input with automatic trigger
        query = st.text_input("💭 What would you like to know?",placeholder=placeholder, key="search_query")
        if query and st.session_state.get('last_query') != query:
            st.session_state['last_query'] = query
            program_name = st.session_state.selected_program if st.session_state.search_mode == "specific" else ""
            with st.spinner("🤔 Searching..."):
                results = get_relevant_chunks(query, program_name=program_name)
                process_search_results(query, results)
            

if __name__ == "__main__":
    main()