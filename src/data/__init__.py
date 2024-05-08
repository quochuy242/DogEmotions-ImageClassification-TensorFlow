import tensorflow as tf
import os
import src
import opendatasets as od
import cv2
import numpy as np
import shutil
import zipfile
import pickle

from tensorflow import keras
from pathlib import Path
from typing import Tuple

import src.logging


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
            src.logging.info("Downloading data...")
            if os.path.exists(self.data_path / "fer2013"):
                shutil.rmtree(self.data_path / "fer2013")
            od.download(self.url_download, self.data_path)
            self.download = False
            src.logging.info("Downloaded data!!")

            # with zipfile.ZipFile(
            #     self.data_path / "fer2013" / "fer2013.zip", "r"
            # ) as zip_ref:
            #     zip_ref.extractall(self.data_path / "fer2013")

        if os.path.exists(self.data_path / "fer2013" / "test" / "disgust"):
            shutil.rmtree(self.data_path / "fer2013" / "test" / "disgust")
            shutil.rmtree(self.data_path / "fer2013" / "train" / "disgust")
        if os.path.exists(self.data_path / "fer2013"):
            shutil.copytree(
                self.data_path / "fer2013" / "test",
                self.data_path,
                dirs_exist_ok=True,
            )
            shutil.copytree(
                self.data_path / "fer2013" / "train",
                self.data_path,
                dirs_exist_ok=True,
            )
        shutil.rmtree(self.data_path / "fer2013")
        src.logging.info("Prepared dataset!!")


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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(
            img, dsize=self.image_size[:-1], interpolation=cv2.INTER_LINEAR
        )
        img = img.astype(np.float32) / 255.0
        img = keras.utils.img_to_array(img)
        img = np.expand_dims(img, axis=-1)
        return img

    def get_dataset(self, path: Path) -> tf.data.Dataset:

        src.logging.info(f"Loading dataset from: {path}")
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
            src.logging.exception(e)

        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        N = len(dataset)
        dataset = dataset.shuffle(buffer_size=N + 1)

        train_size = int(N * self.train_ratio)
        test_size = int(N * self.test_ratio)
        val_size = int(N * self.val_ratio)
        train_ds = dataset.take(train_size)
        test_ds = dataset.skip(train_size).take(test_size)
        val_ds = dataset.skip(train_size + test_size).take(val_size)

        src.logging.info(f"Number of train samples: {len(train_ds)}")
        src.logging.info(f"Number of val samples: {len(val_ds)}")
        src.logging.info(f"Number of test samples: {len(test_ds)}")
        return train_ds, test_ds, val_ds

    def save(self, dataset: tf.data.Dataset, path: str) -> None:
        # Create directory if not exists
        save_dir = self.save_path / path
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save dataset to TFRecord files
        save_dir = str(save_dir)
        dataset.save(save_dir, compression='GZIP')
        with open(save_dir + '/element_spec', 'wb') as out_:
            pickle.dump(dataset.element_spec, out_)
        src.logging.info(f'Saved dataset to: "{save_dir}"')


class DataLoader:
    def __init__(self, config):
        self.load_path = config["load_path"]
        self.batch_size = config["batch_size"]

    def load(self, ds_name: str = "train"):
        match ds_name.lower():
            case "train":
                ds_path = os.path.join(self.load_path, "train")
            case "test":
                ds_path = os.path.join(self.load_path, "test")
            case "val":
                    ds_path = os.path.join(self.load_path, "test")
        
        with open(ds_path + "/element_spec", "rb") as in_:
            es = pickle.load(in_)
        ds = tf.data.Dataset.load(ds_path, es, compression='GZIP')
        ds = ds.batch(batch_size=self.batch_size).cache().prefetch(tf.data.AUTOTUNE)
        return ds
