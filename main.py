#!/usr/bin/env python3
"""
Architecture Assistant - Main Entry Point

This script serves as the main entry point for the Architecture Assistant application.
It provides a command-line interface to run different modes of the application.
"""

import argparse
import sys
import os
from pathlib import Path
import subprocess
import webbrowser
from typing import Optional

def run_streamlit_app(app_file: str, port: int = 8501) -> None:
    """Run a Streamlit application.
    
    Args:
        app_file: The name of the Streamlit app file to run
        port: The port number to run the app on
    """
    try:
        # Construct the full path to the app file
        app_path = str(Path(__file__).parent / "streamlit" / app_file)
        
        if not os.path.exists(app_path):
            print(f"Error: Could not find {app_file} in the streamlit directory.")
            return
            
        # Run the Streamlit app
        cmd = [
            "streamlit", "run", 
            "--server.port", str(port),
            "--server.headless", "false",
            "--server.runOnSave", "true",
            app_path
        ]
        
        print(f"Starting {app_file} on http://localhost:{port}")
        print("Press Ctrl+C to stop the server")
        
        # Open the browser automatically
        webbrowser.open(f"http://localhost:{port}")
        
        # Run the command
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\nShutting down the server...")
        sys.exit(0)
    except Exception as e:
        print(f"Error running {app_file}: {e}")
        sys.exit(1)

def check_dependencies() -> bool:
    """Check if all required dependencies are installed."""
    try:
        import streamlit
        import groq
        import langchain
        import pandas
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install the required dependencies using:")
        print("pip install -r requirements.txt")
        return False

def setup_environment() -> bool:
    """Set up the environment variables."""
    if not os.path.exists('.env'):
        print("No .env file found. Creating one from .env.example...")
        try:
            import shutil
            shutil.copy('.env.example', '.env')
            print("Created .env file. Please update it with your API keys.")
            return False
        except Exception as e:
            print(f"Error creating .env file: {e}")
            return False
    return True

def main() -> None:
    """Main entry point for the Architecture Assistant."""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Architecture Assistant - AI-powered architectural design and budgeting tool"
    )
    
    # Add subcommands for different modes
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Budget mode
    budget_parser = subparsers.add_parser('budget', help='Run the budget assistant')
    budget_parser.add_argument(
        '--port', type=int, default=8501,
        help='Port to run the Streamlit app on (default: 8501)'
    )
    
    # Simple budget mode (new)
    simple_budget_parser = subparsers.add_parser('simple-budget', help='Run the simple budget assistant')
    simple_budget_parser.add_argument(
        '--port', type=int, default=8502,
        help='Port to run the Streamlit app on (default: 8502)'
    )
    
    # Design mode
    design_parser = subparsers.add_parser('design', help='Run the design assistant')
    design_parser.add_argument(
        '--port', type=int, default=8503,
        help='Port to run the Streamlit app on (default: 8503)'
    )
    
    # List modes
    list_parser = subparsers.add_parser('list', help='List available modes')
    
    # Check command line arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Handle list command
    if args.command == 'list':
        print("Available modes:")
        print("  budget - Run the budget assistant")
        print("  design - Run the design assistant")
        return
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Set up environment
    if not setup_environment():
        print("Please update the .env file with your API keys and try again.")
        sys.exit(1)
    
    # Run the appropriate mode
    if args.command == 'budget':
        run_streamlit_app("streamlit_budget_app_fixed.py", args.port)
    elif args.command == 'simple-budget':
        run_streamlit_app("simple_budget_app.py", args.port)
    elif args.command == 'design':
        run_streamlit_app("streamlit_design_app.py", args.port)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()