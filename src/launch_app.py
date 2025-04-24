#!/usr/bin/env python3
"""
Launcher script for Streamlit search app that disables file watcher
to avoid PyTorch conflict errors.
"""
import os
import sys
import subprocess
from pathlib import Path

# Get the absolute path to the search_app.py
current_dir = Path(__file__).parent
app_path = current_dir / "search_app.py"

# Command to run Streamlit with the file watcher disabled
cmd = [
    "streamlit", "run", 
    str(app_path),
    "--server.fileWatcherType", "none"
]

print("Launching Streamlit app with file watcher disabled...")
# Execute the command
subprocess.run(cmd)