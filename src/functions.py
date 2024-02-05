import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('ggplot')
emotions = ['angry', 'happy', 'relaxed', 'sad']

def show_image(paths: list, rows: int, cols: int, figsize: tuple = (15, 15)) -> None:
    """
    Plotting images through list paths
    :param paths: list the path of images
    :param rows: number of row to show
    :param cols: number of col to show
    :param figsize: size of figure, default is (15, 15)
    :return: a gird have rows * cols images
    """
    global emotions
    for i in range(rows + cols):
        plt.subplot(rows, cols, i+1)
        plt.grid(False)
        plt.imshow(paths[i])
        plt.xlabel(emotions)
    plt.show()

