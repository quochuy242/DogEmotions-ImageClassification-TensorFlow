import os
import data.preprocessing as preprocess
import model.build_model as build
import model.evaluate_model as evaluate


def preproces():
    current_path = os.getcwd()
    dataset_path = os.path.join(current_path, 'Dataset')

    # Load Data
    data = preprocess.load_data(dataset_path)
    print(f'Length of Data: {len(data)}')

    # Applying Data Pipeline
    data, after_len_data = preprocess.pipeline(data)
    print(f'After through pipeline, length of data: {len(data)}')

    # Train Validation Splitting
    train, val = preprocess.train_val_split(data, train_rate=0.8)
    print(f'Length of Training Dataset: {len(train)}')
    print(f'Length of Validation Dataset: {len(val)}')
    return train, val

def training(train, val):
    # Building CNN model
    model = build.create_model()
    print(model.summary())
    best_model, history = build.fit_model(model, train=train, val=val)
    return history

def evaluting(history):
    # Show the results of the training
    evaluate.plot_metrics(history)
    

if __name__ == '__main__':
    train_ds, val_ds = preproces()
    history = training(train=train_ds, val=val_ds)
    evaluting(history)


