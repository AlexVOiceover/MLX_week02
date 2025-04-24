import streamlit as st
import sys
import os
from pathlib import Path
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the search functionality directly (since we're in the same directory)
from search import load_query_model, search as run_search

# Use Streamlit's cache_resource decorator to keep the model loaded between searches
@st.cache_resource
def get_cached_model():
    """Cache the model loading to avoid reloading between searches"""
    print("Loading model from W&B (first time only)...")
    return load_query_model()

@st.cache_resource
def get_chromadb_connection():
    """Cache the ChromaDB connection to avoid reconnecting between searches"""
    import chromadb
    import os
    from pathlib import Path
    
    # Check if we should use remote ChromaDB
    use_remote = os.getenv("USE_REMOTE_CHROMA", "False").lower() == "true"
    
    if use_remote:
        # Connect to containerized ChromaDB
        print("Connecting to remote ChromaDB (first time only)...")
        chroma_host = os.getenv("CHROMA_HOST", "localhost")
        chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        return chromadb.HttpClient(host=chroma_host, port=chroma_port)
    else:
        # Use local ChromaDB
        print("Connecting to local ChromaDB (first time only)...")
        chroma_dir = Path(__file__).parent.parent / "data" / "index"
        return chromadb.PersistentClient(path=str(chroma_dir))

# Configure the page
st.set_page_config(page_title="Semantic Search Engine", page_icon="🔍", layout="wide")

# App title and description
st.title("🔍 Semantic Search Engine")
st.markdown(
    """
This app uses a neural network to search through documents based on semantic meaning.
Enter a query and the system will find the most relevant documents.
"""
)

# Load resources once at startup
cached_model_resources = get_cached_model()
cached_client = get_chromadb_connection()

# Input for search query with a key to detect changes
query = st.text_input("Enter your search query:", "", key="search_input")

# Add a search button 
search_button = st.button("Search")

# Track if the query was just submitted via Enter key
query_submitted = False
if "previous_query" not in st.session_state:
    st.session_state.previous_query = ""

# Check if Enter was pressed (query changed but button wasn't clicked)
if query != st.session_state.previous_query:
    query_submitted = True
    st.session_state.previous_query = query

# Function to display search results
def display_results(query):
    with st.spinner("Searching for relevant documents..."):
        # Create a placeholder for results
        results_container = st.container()

        # Capture stdout to show info messages
        import io
        import contextlib
        import sys

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            # Run the search with cached resources
            run_search(query, cached_resources=cached_model_resources, cached_client=cached_client)

        # Parse results from stdout
        output = stdout.getvalue()

        # Display model loading info in small text
        if "Loading model" in output:
            st.info("Loading model and preparing search...")

        # Extract results
        if "Search results for:" in output:
            results_section = output.split("Search results for:")[1]
            result_blocks = results_section.split("-" * 50)[
                1:-1
            ]  # Skip the header and empty last element

            if "No results found" in results_section:
                results_container.warning("No results found. Try a different query.")
                return

            # Display each result in a card
            for i, block in enumerate(result_blocks):
                lines = block.strip().split("\n")

                # Extract info (expected format from search.py)
                doc_id = lines[0].replace("Result", "").strip().split(":")[1].strip()
                similarity = lines[1].split("(")[0].strip()
                text = " ".join(lines[2:]).replace("Text:", "").strip()

                # Create a card for this result
                with results_container.container():
                    st.markdown(f"### Result {i+1}: {doc_id}")
                    st.markdown(f"**Similarity:** {similarity}")
                    st.markdown(f"**Text:** {text}")
                    st.markdown("---")
        else:
            results_container.warning("Search completed but no results were found.")


# Run search when the button is clicked or Enter is pressed
if (search_button or query_submitted) and query:
    display_results(query)
elif search_button and not query:
    st.warning("Please enter a search query first.")

# Display ChromaDB connection info
st.markdown("---")
with st.expander("📊 ChromaDB Connection Information"):
    # Check connection type
    use_remote = os.getenv("USE_REMOTE_CHROMA", "False").lower() == "true"
    chroma_host = os.getenv("CHROMA_HOST", "localhost")
    chroma_port = os.getenv("CHROMA_PORT", "8000")
    
    if use_remote:
        st.info(f"🔌 Connected to remote ChromaDB at {chroma_host}:{chroma_port}")
        
        # If using Docker, try to get container info
        if chroma_host == "localhost" or chroma_host == "127.0.0.1":
            try:
                import subprocess
                container_info = subprocess.check_output(
                    ["docker", "ps", "--filter", "name=chroma", "--format", "{{.Names}} ({{.Image}}) - Running since {{.RunningFor}}"]
                ).decode().strip()
                
                if container_info:
                    st.success(f"🐳 Docker container: {container_info}")
            except:
                st.warning("Docker container information not available")
        
        # Try to get collection info
        try:
            client = cached_client
            collection_names = client.list_collections()
            collection_info = []
            
            for coll in collection_names:
                name = coll.name
                try:
                    count = client.get_collection(name).count()
                    collection_info.append(f"{name} ({count} items)")
                except:
                    collection_info.append(f"{name} (count unavailable)")
            
            if collection_info:
                st.write("📚 Collections:", ", ".join(collection_info))
        except Exception as e:
            st.error(f"Error getting collection info: {str(e)}")
    else:
        chroma_dir = Path(__file__).parent.parent / "data" / "index"
        st.info(f"📂 Using local ChromaDB at {chroma_dir}")

# Footer
st.markdown("---")
st.markdown("Semantic Search Engine | Built with PyTorch, ChromaDB and Streamlit")
