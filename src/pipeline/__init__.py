import src
import os
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from src.data import DataLoader, DataTransformation, DataIngestion
from src.model import MLP, CNN, ViT

config = src.read_yaml(Path("./config/config.yaml"))
ingestion_config = config["data_ingestion"]
transformation_config = config["data_transformation"]
loader_config = config["data_loader"]

params = src.read_yaml(Path("params.yaml"))


class DataPipeline:
    def __init__(self) -> None:

        if os.path.exists(ingestion_config["data_path"]):
            self.download = False
        else:
            self.download = True
        self.ingestion = DataIngestion(config=ingestion_config, download=self.download)
        self.transform = DataTransformation(config=transformation_config)
        self.loader = DataLoader(config=loader_config)

    @property
    def run_pipeline(self):
        src.logging.info(f"Running data pipeline...")
        if self.download:
            self.ingestion.download_data
            train, test, val = self.transform.get_dataset(path=self.ingestion.data_path)
            [
                self.transform.save(ds, path)
                for ds, path in zip([train, test, val], ["train", "test", "val"])
            ]

        return (
            self.loader.load(ds_name="train"),
            self.loader.load(ds_name="test"),
            self.loader.load(ds_name="val"),
        )


class ModelTrainingPipeline:
    def __init__(self, model: str = "CNN") -> None:
        self.model = None
        match model.lower:
            case "cnn":
                self.model = CNN(params=params["CNN"])
            case "mlp":
                self.model = MLP(params=params["MLP"])
            case "vit":
                self.model = ViT(params=params["ViT"])
        if self.model == None:
            self.model = CNN(params=params["CNN"])
        self.model.build
        self.train_ds, self.val_ds = None, None
        self.history = None

    @property
    def run_pipeline(self):
        src.logging.info(f"Running model training pipeline...")
        self.train_ds, _, self.val_ds = DataPipeline().run_pipeline
        self.history = self.model.fit(self.train_ds, self.val_ds)

        for metric in self.history.history.keys():
            if "val" in metric:
                plt.scatter(
                    self.history.epoch,
                    self.history.history[metric],
                    label=metric,
                    color="orange",
                )
            else:
                plt.plot(
                    self.history.epoch,
                    self.history.history[metric],
                    label=metric,
                    color="blue",
                )
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(f"{self.model.model._name}: {metric}")
        plt.savefig(f"visualize/{self.model.model._name}/{metric}.png")
        return self.history, self.model


class ModelEvaluationPipeline:
    def __init__(self, model: str = "CNN") -> None:
        try:
            self.model = keras.models.load_model(f"weights/{model.upper}/best.keras")
        except Exception as e:
            src.logging.exception(e)

    @property
    def run_pipeline(self):
        src.logging.info(f"Running model evaluation pipeline...")
        _, self.test_ds, _ = DataPipeline().run_pipeline
        result = src.model.evaluate(self.model, self.test_ds)
        return result
