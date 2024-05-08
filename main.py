import os
import os.path as osp
import src
import matplotlib.pyplot as plt

from src import data, model
from pathlib import Path

if __name__ == "__main__":
    # Get config and params
    config = src.read_yaml(Path("config/config.yaml"))
    params = src.read_yaml(Path("params.yaml"))

    # Download data
    data_ingestion = data.DataIngestion(
        config=config["data_ingestion"],
        download=(
            not (
                osp.exists(config["data_ingestion"]["data_path"])
                and osp.getsize(config["data_ingestion"]["data_path"]) > 0
            )
        ),
    )
    data_ingestion.download_data

    # Get dataset
    data_transformation = data.DataTransformation(config=config["data_transformation"])
    train_ds, val_ds = data_transformation.get_dataset(
        path=Path(config["data_ingestion"]["data_path"]),
    )

    # Train model
    cnn = model.CNN(params=params["CNN"]).build_model
    history = model.fit_model(model=cnn, train_ds=train_ds, val_ds=val_ds, epochs=10)

    # Show result of training
    for metric in history.history.keys():
        if "val" in metric:
            plt.scatter(
                history.epoch,
                history.history[metric],
                label=metric,
                color="orange",
            )
        else:
            plt.plot(
                history.epoch,
                history.history[metric],
                label=metric,
                color="blue",
            )
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(f"{cnn._name}: {metric}")
        os.makedirs(name=f"visualize/{cnn._name}", exist_ok=True)
        plt.savefig(f"visualize/{cnn._name}/{metric}.png")
