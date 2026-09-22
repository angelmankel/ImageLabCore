"""
Entry point for running Imagelab_Studio_Extension as a module.

Usage:
    python -m Imagelab_Studio_Extension generate --all
    python -m Imagelab_Studio_Extension generate basic_sdxl
    python -m Imagelab_Studio_Extension generate basic_sdxl --dry-run
    python -m Imagelab_Studio_Extension validate --all
    python -m Imagelab_Studio_Extension validate basic_sdxl
    python -m Imagelab_Studio_Extension list
    python -m Imagelab_Studio_Extension watch  # requires: pip install watchdog
"""
import sys
import os

# Add this directory to path so we can import cli directly
# This avoids loading __init__.py which requires ComfyUI's server module
sys.path.insert(0, os.path.dirname(__file__))

from cli import main

if __name__ == "__main__":
    sys.exit(main())
