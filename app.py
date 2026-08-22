import os
import sys

# Add ai/ to python path and import main app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ai"))
import app
