import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('ggplot')
emotions = ['angry', 'happy', 'relaxed', 'sad']
os.chdir('D:/Python Project/DogEmotions-ImageClassification-TensorFlow')
current_path = os.getcwd()
data_path = os.path.join(current_path, 'Dataset')


def show_image(paths: list, rows: int, cols: int, figsize: tuple = (10, 10)) -> None:
    """
    Plotting images through list paths
    :param paths: list the path of images
    :param rows: number of row to show
    :param cols: number of col to show
    :param figsize: size of figure, default is (15, 15)
    :return: a gird have rows * cols images
    """
    global emotions
    plt.figure(figsize=figsize)
    for i in range(rows*cols):
        plt.subplot(rows, cols, i + 1)
        plt.grid(False)
        image = cv2.imread(paths[i])
        plt.imshow(image)
        plt.axis('off')
        for emo in emotions:
            if emo in paths[i]:
                plt.title(emo)
    plt.show()


list_image = [os.path.join(data_path, 'angry', 'angry' + str(i) + '.jpg') for i in range(25)]
print(list_image)

show_image(list_image, rows=5, cols=5)
