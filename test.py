from isoduration import DurationFormattingException
import keras
import os
import warnings
import cv2
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm, trange


def load(filename: Path) -> np.ndarray:
    img = cv2.imread(filename)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def preprocesing(img: np.ndarray) -> np.ndarray:
    img = cv2.resize(img, dsize=(128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    return img


def main():

    # Load models from weights directory
    model = keras.models.load_model("weights/CNN/best.keras")

    # Load images
    print("Loading images...")
    images = []
    classes = os.listdir("dataset/test")
    for classname in tqdm(classes):
        filenames = os.listdir(f"dataset/test/{classname}")
        for filename in tqdm(filenames):
            if filename[-4:] not in [".jpg", ".png"]:
                continue
            path = f"dataset/test/{classname}/{filename}"
            image = load(path)
            image = preprocesing(image)
            images.append(image)

    # Predicting the images
    durations = []
    print("Predicting the images...")
    for image in tqdm(images):
        start = time.time()
        model.predict(image)
        durations.append(time.time() - start)

    # Calc mean of time prediction
    print(f"Mean of time prediction: {np.mean(durations)}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print("_" * 80)
    main()
