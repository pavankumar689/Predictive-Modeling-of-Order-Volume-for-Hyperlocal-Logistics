#!/usr/bin/env python3
"""
Alternative entry point for the Streamlit application
This file can be used as the main entry point for some deployment platforms
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application"""
    port = os.environ.get('PORT', '8501')
    
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
        '--server.port', port,
        '--server.address', '0.0.0.0',
        '--server.headless', 'true'
    ]
    
    print(f"Starting Streamlit on port {port}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()