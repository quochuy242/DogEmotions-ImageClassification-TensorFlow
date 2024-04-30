import src.model
from pathlib import Path

if __name__ == "__main__":
    params = src.read_yaml(Path("params.yaml"))
    cnn = src.model.CNN(params=params["CNN"]).build
