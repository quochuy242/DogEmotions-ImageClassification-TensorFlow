import tensorflow as tf
import os
import opendatasets as od
import shutil
import os.path as osp

from pathlib import Path
from src import logging


class DataIngestion:
    def __init__(
        self,
        config: dict,
        download: bool = False,
    ) -> None:
        self.url_download = config["url_download"]
        self.data_path = Path(config["data_path"])
        self.download = download
        return

    @property
    def download_data(self):
        if self.download:
            logging.info("Downloading data...")
            if osp.exists(self.data_path / "fer2013"):
                shutil.rmtree(self.data_path / "fer2013")
            od.download(self.url_download, self.data_path)
            self.download = False
            logging.info("Downloaded data!!")

        if osp.exists(self.data_path / "fer2013" / "test" / "disgust"):
            shutil.rmtree(self.data_path / "fer2013" / "test" / "disgust")
            shutil.rmtree(self.data_path / "fer2013" / "train" / "disgust")

        if osp.exists(self.data_path / "fer2013"):
            shutil.move(self.data_path / "fer2013" / "test", self.data_path)
            shutil.move(self.data_path / "fer2013" / "train", self.data_path)
            shutil.rmtree(self.data_path / "fer2013")

        logging.info("Prepared dataset!!")


class DataTransformation:
    def __init__(self, config: dict) -> None:
        self.image_size = (
            config["image_height"],
            config["image_width"],
        )
        self.image_channels_expactation = config["image_channels_expectation"]
        self.class_names = []
        self.batch_size = config["batch_size"]

    def get_dataset(self, path: Path) -> tuple[tf.data.Dataset]:

        logging.info(f"Loading dataset from: {path}")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            path / "train",
            seed=242,
            image_size=self.image_size,
            batch_size=self.batch_size,
            shuffle=True,
            label_mode="categorical",
            color_mode="grayscale" if self.image_channels_expactation == 1 else "rgb",
            interpolation="gaussian",
            validation_split=0.1,
            subset="training",
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            path / "test",
            seed=242,
            image_size=self.image_size,
            batch_size=self.batch_size,
            shuffle=True,
            label_mode="categorical",
            color_mode="grayscale" if self.image_channels_expactation == 1 else "rgb",
            interpolation="gaussian",
            validation_split=0.1,
            subset="validation",
        )
        self.class_name = train_ds.class_names

        for image, label in train_ds:
            logging.info(f"Batched image shape: {image.shape}")
            logging.info(f"Batched label shape: {label.shape}")
            break

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        return train_ds, val_ds
