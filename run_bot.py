#!/usr/bin/env python
"""Simple launcher for the trading bot"""
import sys
import subprocess

if __name__ == "__main__":
    # Import and run the main launch function
    try:
        from pocket_option_bot import launch
        print("Starting Trading Bot on localhost:5000...")
        print("Open your browser to: http://localhost:5000")
        launch()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
