# pyright: standard
import sys
import os
from os import environ

home = environ.get("HOME")
run_folder = os.path.dirname(os.path.abspath(__file__))

# Add parent directory to path for sibling imports
thought_virus_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(thought_virus_dir, "src"))