import streamlit as st
import sys
from pathlib import Path
import torch

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
    from pathlib import Path
    
    print("Connecting to ChromaDB (first time only)...")
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

# Input for search query
query = st.text_input("Enter your search query:", "")

# Add a search button
search_button = st.button("Search")


# Load resources once at startup
cached_model_resources = get_cached_model()
cached_client = get_chromadb_connection()

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


# Run search when the button is clicked
if search_button and query:
    display_results(query)
elif search_button:
    st.warning("Please enter a search query first.")

# Footer
st.markdown("---")
st.markdown("Semantic Search Engine | Built with PyTorch, ChromaDB and Streamlit")
