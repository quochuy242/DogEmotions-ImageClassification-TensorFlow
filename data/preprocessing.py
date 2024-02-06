import tensorflow as tf
import os

Dataset = tf.data.Dataset
AUTOTUNE = tf.data.AUTOTUNE


def load_data(data_path: str) -> Dataset:
    """
    Loading dataset from input path
    :param data_path:
    :return: data
    """
    global emotions
    angry_images = Dataset.list_files(os.path.join(data_path, emotions[0]) + '/*.jpg')
    happy_images = Dataset.list_files(os.path.join(data_path, emotions[1]) + '/*.jpg')
    relaxed_images = Dataset.list_files(os.path.join(data_path, emotions[2]) + '/*.jpg')
    sad_images = Dataset.list_files(os.path.join(data_path, emotions[3]) + '/*.jpg')

    angry_with_label = Dataset.zip((angry_images, Dataset.from_tensor_slices(tf.ones(len(angry_images)))))
    happy_with_label = Dataset.zip((happy_images, Dataset.from_tensor_slices(2 * tf.ones(len(angry_images)))))
    relaxed_with_label = Dataset.zip((relaxed_images, Dataset.from_tensor_slices(3 * tf.ones(len(angry_images)))))
    sad_with_label = Dataset.zip((sad_images, Dataset.from_tensor_slices(4 * tf.ones(len(angry_images)))))

    data = angry_with_label.concatenate(happy_with_label).concatenate(relaxed_with_label).concatenate(sad_with_label)
    return data


def train_val_split(data: Dataset, train_rate: float = 0.8):
    train_ds = data.take(int(len(data) * train_rate))
    val_ds = data.skip(int(len(data) * train_rate)).take(int(len(data) * (1 - train_rate)))

    train_ds = train_ds[..., tf.newaxis].astype('float32')
    val_ds = val_ds[..., tf.newaxis].astype('float32')

    return train_ds, val_ds


def pipeline(data: Dataset):
    """
    Building pipeline for data. Help increasing efficiency of model
    :param data: tf.data.Dataset
    :return: data after passing pipeline and length of new data
    """
    data = data.cache()
    data = data.shuffle(buffer_size=10000)  # buffer_size should be more length of data
    data = data.batch(128)
    data = data.prefetch(AUTOTUNE)

    return data, len(data)
