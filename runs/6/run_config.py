# pyright: standard
import os
from os import environ
from pathlib import Path

home = environ.get("HOME")
run_folder = os.path.dirname(os.path.abspath(__file__))
shared_folder = (Path(run_folder).parent.parent / "shared").as_posix()