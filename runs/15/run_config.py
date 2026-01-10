# pyright: standard
import os
from os import environ

home = environ.get("HOME")
run_folder = os.path.dirname(os.path.abspath(__file__))