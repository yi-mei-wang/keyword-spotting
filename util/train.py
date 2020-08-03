import json

import numpy as np
import tensorflow.keras as keras

from sklearn.model_selection import train_test_split

from constants import DATA_JSON_PATH, SAVED_MODEL_PATH

LEARNING_RATE = 0.0001
EPOCHS = 40  # epoch = number of times the network will see the whole dataset for training purposes
BATCH_SIZE = 32  # number of samples the network will feed before running the back-propagation algorithm

NUM_KEYWORDS = 10  # corresponds to the number of mappings in data.json (e.g., 'on', 'down')


def load_dataset(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    # extract inputs and targets
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])

    return X, y


def get_data_splits(data_path, test_size=0.1, test_validation=0.1):
    # load the dataset from data_path
    X, y = load_dataset(data_path)

    # create train/val/test splits
    # validation - hyperparameter tuning, test - for after everything is done
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=test_validation)

    # convert input from 2d to 3d arrays using np.newaxis
    # (no. of segments, no. of coefficients=13)
    # like spreading the array and then add a new dimensional
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape, learning_rate, error="sparse_categorical_crossentropy"):
    # build network - NN that has sequential layers
    model = keras.Sequential()

    # conv layer 1
    # 64 is the number of filters, (3,3) is the shape?
    model.add(keras.layers.Conv2D(64, (3, 3), activation="relu",
                                  input_shape=input_shape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)))  # used to target overfitting
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))

    # conv layer 2
    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.001)))  # used to target overfitting
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))

    # conv layer 3
    model.add(keras.layers.Conv2D(32, (2, 2), activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.001)))  # used to target overfitting
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))

    # flatten the output of the convolutional layers and feed it into a dense
    # layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))  # shuts down 30% of the neurones in
    # the dense layer during training to help with overfitting - network will
    # adapt such that all neurones will take equal responsibility to do
    # classification

    # softmax classifier - to decide how to pick the different predictions
    model.add(keras.layers.Dense(NUM_KEYWORDS, activation="softmax"))  # returns
    # [] with values and scores for the keywords, e.g. [0.1, 0.7, 0.2] ->
    # probability of the audio file being each of the keywords - should add
    # up to 1

    # compile the model
    optimiser = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimiser, loss=error, metrics=["accuracy"])

    # print model overview
    model.summary()

    return model


def main():
    # load train/validation/test data splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits(DATA_JSON_PATH)

    # build the CNN model - keyword spotting system
    # CNN takes in a 3d array
    # (# segments, # coefficients = 13, 1 (fundamental for CNN - depth of information))
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape, LEARNING_RATE)

    # train the model - fit() comes from the keras api
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_validation, y_validation))

    # evaluate the model
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test error: {test_error}, test accuracy: {test_accuracy}")

    # save the model
    model.save(SAVED_MODEL_PATH)


if __name__ == "__main__":
    print("running train.py")
    main()
