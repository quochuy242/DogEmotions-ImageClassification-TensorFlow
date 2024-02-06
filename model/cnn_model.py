import tensorflow as tf
from tensorflow_addons.metrics import F1Score
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Resizing, MaxPooling2D, Input, Resizing, Rescaling
from keras.metrics import Recall, Precision, Accuracy
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy

Dataset = tf.data.Dataset
AUTOTUNE = tf.data.AUTOTUNE

early_stopping = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, patience=100)
model_checkpoint = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss',
                                   save_best_only=True, mode='min')


def lr_time_based_decay(epoch, lr):
    initial_lr = 0.0001
    epochs = 1000
    decay = initial_lr / epochs
    return lr * 1 / (1 + decay * epoch)


def build_model(resize_shape: tuple = (32, 32), initial_learning_rate: float = 0.0001):
    model = Sequential()
    model.add(Rescaling(1. / 255, input_shape=(384, 384, 3)))
    model.add(Resizing(resize_shape[0], resize_shape[1]))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(units=4))

    model.compile(optimizer=Adam(learning_rate=initial_learning_rate), loss=CategoricalCrossentropy(),
                  metrics=[Recall(), Precision(), F1Score(), Accuracy()])

    return model


def fit_model(model, epochs, train, val):
    history = model.fit(train,
                        validation_data=val,
                        epochs=epochs,
                        verbose=50,
                        callbacks=[early_stopping, model_checkpoint,
                                   LearningRateScheduler(schedule=lr_time_based_decay, verbose=1)])
    model = tf.keras.models.load_model('best_model.h5')
    return model, history
