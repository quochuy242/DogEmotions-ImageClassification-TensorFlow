import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import Sequential, layers, optimizers, losses, metrics, callbacks
from pathlib import Path
from src import logging, read_yaml
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

params = read_yaml(Path("params.yaml"))

data_augmentation = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
]

early_stop = callbacks.EarlyStopping(
    monitor=params["early_stopping"]["monitor"],
    patience=params["early_stopping"]["patience"],
    verbose=0,
    mode="auto",
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor=params["reduce_lr"]["monitor"],
    factor=params["reduce_lr"]["factor"],
    patience=params["reduce_lr"]["patience"],
    min_lr=params["reduce_lr"]["min_lr"],
    mode="auto",
    verbose=0,
)


class CNN:
    def __init__(self, params) -> None:
        self.input_shape = params["input_shape"]
        self.num_classes = params["num_classes"]
        self.conv_units = params["conv_units"]
        self.dense_units = params["dense_units"]
        self.dropout_rate = params["dropout_rate"]
        self.initial_lr = params["learning_rate"]
        self.l1 = params["l1"]
        self.l2 = params["l2"]
        self.epochs = params["epochs"]
        self.model = None

    @property
    def build(self) -> Sequential:
        model = Sequential(data_augmentation)

        for units in self.conv_units:
            model.add(layers.Conv2D(units, 3, activation=tf.nn.relu, padding="same"))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Flatten())
        for units in self.dense_units:
            model.add(layers.Dense(units, activation=tf.nn.relu))

        model.add(layers.Dropout(self.dropout_rate))
        model.add(
            layers.Dense(
                self.num_classes,
                activation=tf.nn.softmax,
                name="output",
                kernel_regularizer=tf.keras.regularizers.l1(0.004),
                activity_regularizer=tf.keras.regularizers.l2(0.004),
            ),
        )
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.initial_lr),
            loss=losses.CategoricalCrossentropy(),
            metrics=[
                metrics.CategoricalAccuracy(),
                metrics.AUC(),
                metrics.Precision(),
                metrics.Recall(),
                metrics.F1Score(average="weighted"),
            ],
        )
        model._name = "CNN"
        logging.info(f"{model._name} model summary: ")
        logging.info(model.summary())
        keras.utils.plot_model(
            model,
            to_file=f"visualize/{model._name}.png",
            show_shapes=True,
            show_layer_names=True,
        )
        self.model = model

    def fit(self, train_ds, val_ds) -> callbacks.History:
        return self.model.fit(
            train_ds,
            epochs=self.epochs,
            validation_data=val_ds,
            verbose=self.epochs // 10,
            callbacks=[
                early_stop,
                reduce_lr,
                callbacks.ModelCheckpoint(
                    filepath=f"weights/{self.model._name}/best.keras",
                    monitor="val_f1_score",
                    save_best_only=True,
                    verbose=0,
                ),
                callbacks.ModelCheckpoint(
                    filepath=f"weights/{self.model._name}/last.h5",
                    monitor="val_loss",
                    verbose=0,
                ),
                callbacks.CSVLogger("logs/" + self.model._name + ".log"),
            ],
        )

    def evaluate(self, test_ds):
        result = self.model.evaluate(test_ds)
        logging.info(f"Evaluating {self.model._name} model: {result}")

        y_pred = self.model.predict(test_ds)
        y_pred = y_pred.argmax(axis=1)

        y_true = test_ds.map(lambda x, y: y)
        ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot()
        plt.savefig(f"visualize/{self.model._name}/confusion_matrix.png")


class MLP:
    def __init__(self, params) -> None:
        self.input_shape = params["input_shape"]
        self.num_classes = params["num_classes"]
        self.dense_units = params["dense_units"]
        self.dropout_rate = params["dropout_rate"]
        self.initial_lr = params["learning_rate"]
        self.l1 = params["l1"]
        self.l2 = params["l2"]
        self.epochs = params["epochs"]
        self.model = None

    @property
    def build(self) -> Sequential:
        model = Sequential(data_augmentation)
        for units in self.dense_units:
            model.add(layers.Dense(units, activation=tf.nn.relu))
        model.add(layers.Dropout(self.dropout_rate))
        model.add(
            layers.Dense(
                self.num_classes,
                activation=tf.nn.softmax,
                name="output",
                kernel_regularizer=tf.keras.regularizers.l1(0.004),
                activity_regularizer=tf.keras.regularizers.l2(0.004),
            ),
        )
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.initial_lr),
            loss=losses.CategoricalCrossentropy(),
            metrics=[
                metrics.CategoricalAccuracy(),
                metrics.AUC(),
                metrics.Precision(),
                metrics.Recall(),
                metrics.F1Score(average="weighted"),
            ],
        )
        model._name = "MLP"
        logging.info(f"{model._name} model summary: ")
        logging.info(model.summary())
        keras.utils.plot_model(
            model,
            to_file=f"visualize/{model._name}.png",
            show_shapes=True,
            show_layer_names=True,
        )
        self.model = model

    def fit(self, train_ds, val_ds) -> callbacks.History:
        return self.model.fit(
            train_ds,
            epochs=self.epochs,
            validation_data=val_ds,
            verbose=self.epochs // 10,
            callbacks=[
                early_stop,
                reduce_lr,
                callbacks.ModelCheckpoint(
                    filepath=f"weights/{self.model._name}/best.keras",
                    monitor="val_f1_score",
                    save_best_only=True,
                    verbose=0,
                ),
                callbacks.ModelCheckpoint(
                    filepath=f"weights/{self.model._name}/last.h5",
                    monitor="val_loss",
                    verbose=0,
                ),
                callbacks.CSVLogger("logs/" + self.model._name + ".log"),
            ],
        )

    def evaluate(self, test_ds):
        result = self.model.evaluate(test_ds)
        logging.info(f"Evaluating {self.model._name} model: {result}")

        y_pred = self.model.predict(test_ds)
        y_pred = y_pred.argmax(axis=1)

        y_true = test_ds.map(lambda x, y: y)
        ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot()
        plt.savefig(f"visualize/{self.model._name}/confusion_matrix.png")
        return result


class ViT:
    def __init__(self, params) -> None:
        pass
