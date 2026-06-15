"""
Shared path constants for all ML modules.
Import BASE_DIR, ONET_DIR, MODEL_DIR from here instead of repeating
os.path.dirname(…) boilerplate in every file.
"""
import os

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONET_DIR  = os.path.join(BASE_DIR, "data", "onet")
MODEL_DIR = os.path.join(BASE_DIR, "models")