# Semantic Search App

A simple Streamlit app that provides a web interface to the neural semantic search engine.

## Running the App

1. Make sure you have all requirements installed:
   ```
   pip install -r ../requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run search_app.py
   ```

3. The app will open in your default web browser, typically at `http://localhost:8501`

## Features

- Simple, clean interface
- Real-time semantic search
- Displays results with similarity scores
- Shows document previews

## Requirements

This app requires that you have already:
1. Trained the search model
2. Indexed your documents using the indexer

The app leverages the same search functionality used in the command-line version.