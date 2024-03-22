import tensorflow as tf
import os
import cv2
import numpy as np

Dataset = tf.data.Dataset
AUTOTUNE = tf.data.AUTOTUNE
emotions = ['angry', 'happy', 'relaxed', 'sad']


def load_data(data_path: str) -> Dataset:
    """
    Loading dataset from input path
    :param data_path:
    :return: data
    """
    global emotions
    angry_images = Dataset.list_files(os.path.join(data_path, emotions[0]) + '\*.jpg')
    happy_images = Dataset.list_files(os.path.join(data_path, emotions[1]) + '\*.jpg')
    relaxed_images = Dataset.list_files(os.path.join(data_path, emotions[2]) + '\*.jpg')
    sad_images = Dataset.list_files(os.path.join(data_path, emotions[3]) + '\*.jpg')

    angry_with_label = Dataset.zip(
        (angry_images, Dataset.from_tensor_slices([one_hot_encoding(emotions[0]) for _ in range(len(angry_images))])))
    happy_with_label = Dataset.zip(
        (happy_images, Dataset.from_tensor_slices([one_hot_encoding(emotions[1]) for _ in range(len(happy_images))])))
    relaxed_with_label = Dataset.zip((relaxed_images, Dataset.from_tensor_slices(
        [one_hot_encoding(emotions[2]) for _ in range(len(relaxed_images))])))
    sad_with_label = Dataset.zip(
        (sad_images, Dataset.from_tensor_slices([one_hot_encoding(emotions[3]) for _ in range(len(sad_images))])))

    data = angry_with_label.concatenate(happy_with_label).concatenate(relaxed_with_label).concatenate(sad_with_label)
    return data


def train_val_split(data: Dataset, train_rate: float = 0.8):
    train_ds = data.take(int(len(data) * train_rate))
    val_ds = data.skip(int(len(data) * train_rate)).take(len(data) - len(train_ds))
    return train_ds, val_ds


def one_hot_encoding(emotion):
    res = []
    match emotion:
        case 'angry':
            res = [1.0, 0.0, 0.0, 0.0]
        case 'happy':
            res = [0.0, 1.0, 0.0, 0.0]
        case 'relaxed':
            res = [0.0, 0.0, 1.0, 0.0]
        case 'sad':
            res = [0.0, 0.0, 0.0, 1.0]
    return res


def preprocess(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.expand_dims(image, axis=-1)
    return image, label


def pipeline(data: Dataset):
    """
    Building pipeline for data. Help increasing efficiency of model
    :param data: tf.data.Dataset
    :return data, len(data): data after passing pipeline and length of new data
    """
    # new_data = np.array()
    # for i in data:
    #     file_path, label = i.as_numpy_iterator().next()
    #     image, label = preprocess(image, label)
    #     new_data = np.concatenate((new_data, Dataset.zip(image, label)))
    data = data.map(preprocess)
    data = data.cache()
    data = data.shuffle(buffer_size=20000)  # buffer_size should be more length of data
    data = data.batch(64)
    data = data.prefetch(AUTOTUNE)

    return data, len(data)
