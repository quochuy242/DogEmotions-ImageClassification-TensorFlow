import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('ggplot')

def show_image(paths: list, rows: int, cols: int, figsize: tuple = (15, 15)) -> None:
    """
    Plotting images through list paths
    :param paths: list the path of images
    :param rows: number of row to show
    :param cols: number of col to show
    :param figsize: size of figure, default is (15, 15)
    :return: a gird have rows * cols images
    """
    for i in range(1, rows + cols):
        # plt.subplot(rows, cols, i)
        cv2.imshow('image', cv2.imread(paths[i-1]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # plt.show()

os.chdir('D:\Python Project\DogEmotions-ImageClassification-TensorFlow')
current_path = os.getcwd()
data_path = os.path.join(current_path, 'Dataset')
emotions = os.listdir(data_path)

angry_images = os.listdir(os.path.join(data_path, emotions[0]))

show_image(angry_path, 1, 2)
