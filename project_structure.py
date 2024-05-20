import os
import src
from pathlib import Path

import src.logging


list_of_files = [
    f"src/__init__.py",
    f"src/pipeline/__init__.py",
    f"src/data/__init__.py",
    f"src/logging/__init__.py",
    f"src/model/__init__.py",
    f"config.yaml",
    f"params.yaml",
    f"app.py",
    f"main.py",
    f"Dockerfile",
    f"requirements.txt",
    f"research/trials.ipynb",
    f"visualize/",
    f"templates/index.html",
]

for filepath in list_of_files:
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        src.logging.info(f"Creating a new file: {filedir}/{filepath}")
    else:
        src.logging.info(f"{filedir}/{filepath} already exists. Skipping...")
