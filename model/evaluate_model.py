import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
import random
import cv2

os.chdir('D:\Python Project\DogEmotions-ImageClassification-TensorFlow')
from data.preprocessing import load_data, preprocess

emotions = ['angry', 'happy', 'relaxed', 'sad']

def plot_metrics(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    precision = history.history['precision']
    val_pre = history.history['val_precision']
    recall = history.history['recall']
    val_recall = history.history['val_recall']

    epochs = range(len(acc))
    plt.figure(figsize=(10, 6))

    figure, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].plot(epochs, acc, 'red', label='Training Accuracy')
    ax[0, 0].plot(epochs, val_acc, 'blue', label='Validation Accuracy')
    ax[0, 0].set_title('Accuracy')

    ax[0, 1].plot(epochs, loss, 'red', label='Training Loss')
    ax[0, 1].plot(epochs, val_loss, 'blue', label='Validation Loss')
    ax[0, 1].set_title('Loss')

    ax[1, 0].plot(epochs, precision, 'red', label='Training Precision Score')
    ax[1, 0].plot(epochs, val_pre, 'blue', label='Validation Precision Score')
    ax[1, 0].set_title('Precision')

    ax[1, 1].plot(epochs, recall, 'red', label='Training Recall Score')
    ax[1, 1].plot(epochs, val_recall, 'blue', label='Validation Recall Score')
    ax[1, 1].set_title('Recall')

    plt.title('Visualizing the results of the training model')
    plt.legend(loc=0)
    plt.show()
    plt.savefig('plot_metrics.png')

def testing(test_filepath):
    test_ds = load_data(test_filepath)
    random_test = random.choices(test_ds, k=12)
    model = tf.keras.models.load_model('best_model.h5')
    fig, axs = plt.subplots(3, 4, figsize=(10, 6))

    # for r in range(3):
    #     for c in range(4):
    #         image = cv2.imread(random_test[i][])
    #         y_pred = model.predict(image)[0]
    pass


