import tensorflow as tf
import os
import logging as log
import src
import opendatasets as od
import cv2
import numpy as np
import shutil

from tensorflow import keras
from pathlib import Path
from typing import Tuple
from src import logging as log


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
            log.info("Downloading data...")
            od.download(self.url_download, self.data_path)
            self.download = False

            shutil.copy(self.data_path / "fer2013" / "test", self.data_path)
            shutil.copy(self.data_path / "fer2013" / "train", self.data_path)
            shutil.rmtree(self.data_path / "fer2013")


class DataTransformation:
    def __init__(self, config: dict) -> None:
        self.image_size = (
            config["image_height"],
            config["image_width"],
            config["image_channels"],
        )
        self.num_classes = config["num_classes"]
        self.train_ratio = config["train_ratio"]
        self.test_ratio = config["test_ratio"]
        self.val_ratio = config["val_ratio"]
        self.save_path = Path(config["save_path"])

    def get_label(self, path: Path) -> str:
        return path.parent.name

    def one_hot_encode(self, label: str) -> int:
        map_label = {
            "angry": 0,
            "fear": 1,
            "happy": 2,
            "neutral": 3,
            "sad": 4,
            "surprise": 5,
        }
        label = map_label[label]
        return tf.one_hot(label, depth=6)

    def get_image(self, path: Path) -> np.ndarray[float]:
        img = cv2.imread(str(path))
        if self.image_size[-1] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(
            img, dsize=self.image_size[:-1], interpolation=cv2.INTER_LINEAR
        )
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        return img

    def get_dataset(self, path: Path) -> tf.data.Dataset:

        log.info(f"Loading dataset from: {path}")
        list_labels = os.listdir(path)

        list_images = []
        try:
            for label in list_labels:
                label_dir = path / label
                path_images = [
                    label_dir / f for f in os.listdir(label_dir) if f.endswith(".jpg")
                ]
                list_images.extend(path_images)
            images = list(map(self.get_image, list_images))
            labels = list(map(self.get_label, list_images))
            labels = list(map(self.one_hot_encode, labels))
        except Exception as e:
            log.exception(e)

        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        N = len(dataset)
        dataset = dataset.shuffle(buffer_size=N + 1)

        train_size = int(N * self.train_ratio)
        test_size = int(N * self.test_ratio)
        val_size = int(N * self.val_ratio)
        train_ds = dataset.take(train_size)
        test_ds = dataset.skip(train_size).take(test_size)
        val_ds = dataset.skip(train_size + test_size).take(val_size)
        log.info(f"Number of train samples: {len(train_ds)}")
        log.info(f"Number of val samples: {len(val_ds)}")
        log.info(f"Number of test samples: {len(test_ds)}")
        return train_ds, test_ds, val_ds

    def save(self, dataset: tf.data.Dataset, path: str) -> None:
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
        dataset.save(str(self.save_path / path), compression="GZIP")
        log.info(f'Saved dataset to: "{self.save_path}"')


class DataLoader:
    def __init__(self, config):
        self.load_path = config["load_path"]
        self.batch_size = config["batch_size"]

    def load(self, ds_name: str = "train"):
        match ds_name.lower():
            case "train":
                ds = tf.data.Dataset.load(os.path.join(self.load_path, "train"))
            case "test":
                ds = tf.data.Dataset.load(os.path.join(self.load_path, "test"))
            case "val":
                ds = tf.data.Dataset.load(os.path.join(self.load_path, "val"))

        ds = ds.batch(batch_size=self.batch_size).cache().prefetch(tf.data.AUTOTUNE)
        return ds
