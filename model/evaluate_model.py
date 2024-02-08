import matplotlib.pyplot as plt

def plot_metrics(history):

  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  f1_score = history.history['f1_score']
  val_f1_score = history.history['val_f1_score']

  epochs = range(len(acc))
  plt.figure(figsize=(10, 6))

  plt.plot(epochs, acc, 'red', label='Training Accuracy')
  plt.plot(epochs, val_acc, 'blue', label='Validation Accuracy')
  plt.plot(epochs, loss, 'green', label='Training Loss')
  plt.plot(epochs, val_loss, 'orange', label='Validation Loss')
  plt.plot(epochs, f1_score, 'yellow', label='F1-Score')
  plt.plot(epochs, val_f1_score, 'purple', label='Validation F1-Score')


  plt.title('Traing and Validation; Accuracy, Loss and F1-Score')
  plt.legend(loc=0)
  plt.show()

def evaluate(history):
  pass