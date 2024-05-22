import tensorflow as tf
import keras
import os
import warnings
import cv2
import numpy as np
import time
import src
from pathlib import Path
from tqdm import tqdm, trange
from src import logging

osp = os.path


def load_image_file(path, **kwargs):
    try:
        img = keras.utils.load_img(path, **kwargs)
    except Exception as e:
        logging.exception(e)
    arr = keras.utils.img_to_array(img)
    arr = arr / 255.0

    if len(arr.shape) == 2:
        arr = np.expand_dims(arr, axis=-1)
    arr = np.array([arr])
    return arr


def load_image(path: str, **kwargs):
    if osp.isfile(path):
        return load_image_file(path, **kwargs)
    elif osp.isdir(path):
        arrs = []
        file = os.listdir(path)
        for f in file:
            if f[-4:] not in [".jpg", ".png"]:
                continue
            else:
                arr = load_image_file(f"{path}/{f}", **kwargs)
                arrs.append(arr)

        return np.array(arrs)
    else:
        src.logging.error(f"{path} is not a file or directory")
        return None


def main():
    # Load models from weights directory
    model = keras.models.load_model("weights/CNN/best.keras")
    images = load_image("dataset/test", color_mode="grayscale")

    print(images.shape)
    # # Predicting the images
    # durations = []
    # print("Predicting the images...")
    # for image in tqdm(images):
    #     start = time.time()
    #     model.predict(image)
    #     durations.append(time.time() - start)

    # # Calc mean of time prediction
    # print(f"Mean of time prediction: {np.mean(durations)}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print("_" * 80)
    main()
