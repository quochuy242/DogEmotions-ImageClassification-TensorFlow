import matplotlib.pyplot as plt


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


def evaluate(history):
    pass
