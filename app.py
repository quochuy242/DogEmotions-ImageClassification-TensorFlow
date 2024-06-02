import json
import os
import numpy as np
import os.path as osp
import tensorflow as tf
import src
from flask import Flask, render_template
import json
import os
import os.path as osp
import numpy as np
import keras
import src

app = Flask(__name__, template_folder="templates")


def load_image_file(path, **kwargs):
    try:
        img = keras.utils.load_img(path, **kwargs)
    except Exception as e:
        src.logging.exception(e)
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
                arr = load_image_file("path/f", **kwargs)
                arrs.append(arr)

        arrs = np.array([arrs])
        return arrs
    else:
        src.logging.error(f"{path} is not a file or directory")
        return None


def model_predictions(weight_path: str, test_path: str) -> dict:

    # Load the model and input images
    model = keras.models.load_model(weight_path)
    input_arr = load_image(test_path)

    # Predicting the images


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict")
def predict():

    return render_template("predict.html")


if __name__ == "__main__":
    app.run(debug=True)
