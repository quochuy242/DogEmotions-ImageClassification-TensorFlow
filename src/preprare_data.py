import os

os.chdir('D:/Python Project/DogEmotions-ImageClassification-TensorFlow')
current_path = os.getcwd()
data_path = os.path.join(current_path, 'Dataset')
emotions = os.listdir(data_path)


def rename_file(source, destination):
    try:
        os.rename(src=source, dst=destination)
        print(f'File {source} successfully rename to {destination}')
    except FileNotFoundError:
        print(f'Error: File {source} is not exist')
    except PermissionError:
        print(f'Error: Don\'t have permission for renaming file {source}')
    except OSError as e:
        print(f'OS Error: {e}')


for emo in emotions:
    path = os.path.join(data_path, emo)
    for index, image in enumerate(os.listdir(path)):
        rename_file(source=os.path.join(path, image),
                    destination=os.path.join(path, emo + str(index) + '.jpg'))
