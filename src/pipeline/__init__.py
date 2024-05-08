import src
import os
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from src.data import DataLoader, DataTransformation, DataIngestion
import src.logging
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

        if not os.path.exists(self.transform.save_path):
            train, test, val = self.transform.get_dataset(path=self.ingestion.data_path)
            for ds, name in [(train, "train"), (test, "test"), (val, "val")]:
                self.transform.save(dataset=ds, path=name)
            src.logging.info(f"Saved dataset to: {self.ingestion.data_path}")
        else:
            src.logging.info(f"Dataset already exists at: {self.transform.save_path}")
        train = self.loader.load(ds_name="train")
        test = self.loader.load(ds_name="test")
        val = self.loader.load(ds_name="val")
        src.logging.info(f'Loaded dataset from: "{self.loader.load_path}"')
        src.logging.info(f"Number of train samples: {len(train)}")
        src.logging.info(f"Number of val samples: {len(val)}")
        src.logging.info(f"Number of test samples: {len(test)}")
        return train, test, val


class ModelTrainingPipeline:
    def __init__(self, model: str = "CNN") -> None:
        match model.upper():
            case "CNN":
                self.model = CNN(params=params["CNN"])
            case "MLP":
                self.model = MLP(params=params["MLP"])
            case "VIT":
                self.model = ViT(params=params["ViT"])

        self.model.build_model
        self.train_ds, self.test_ds, self.val_ds = DataPipeline().run_pipeline
        self.history = None

    @property
    def run_pipeline(self):
        src.logging.info(f"Running model training pipeline...")

        self.history = self.model.fit_model(self.train_ds, self.val_ds)

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
