import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")

ROOT_DIR = Path(__file__).resolve().parent
list_of_files = [
    f"src/__init__.py",
    f"src/pipeline/__init__.py",
    f"src/training/__init__.py",
    f"src/testing/__init__.py",
    f"src/data/__init__.py",
    f"src/logging/__init__.py",
    f"config/config.yaml",
    f"params.yaml",
    "app.py",
    "main.py",
    "Dockerfile",
    "requirements.txt",
    "research/trials.ipynb",
]

list_of_files = [Path(ROOT_DIR / file) for file in list_of_files]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists!")
