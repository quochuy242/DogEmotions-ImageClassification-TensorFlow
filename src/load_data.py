import tensorflow as tf
import os

Dataset = tf.data.Dataset
AUTOTUNE = tf.data.AUTOTUNE

emotions = ['angry', 'happy', 'relaxed', 'sad']
os.chdir('D:/Python Project/DogEmotions-ImageClassification-TensorFlow')
current_path = os.getcwd()
data_path = os.path.join(current_path, 'Dataset')

angry_imgs = Dataset.list_files()
