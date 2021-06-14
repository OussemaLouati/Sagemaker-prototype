import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adadelta


class Trainer(object):
    def __init__(self):
        pass

    def load_data(self, training_path, validation_path):
        self.training_data = pd.read_csv(training_path).to_numpy()
        self.validation_data = pd.read_csv(validation_path).to_numpy()
        return self.training_data, self.validation_data

    def split_data(self):
        self.x_train, self.y_train = (
            self.training_data[:, 0:8],
            self.training_data[:, 8],
        )
        self.x_val, self.y_val = (
            self.validation_data[:, 0:8],
            self.validation_data[:, 8],
        )
        return self.x_train, self.y_train, self.x_val, self.y_val

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(8, input_dim=8, activation="relu"))
        self.model.add(Dense(16, activation="relu"))
        self.model.add(Dense(16, activation="relu"))
        self.model.add(Dense(8, activation="relu"))
        self.model.add(Dense(1, activation="sigmoid"))
        return self.model

    def compile_model(self, learning_rate, decay):
        optimizer = Adadelta(lr=learning_rate, decay=decay)
        self.model.compile(
            loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )
        return self.model

    def fit_model(self, batch_size, epochs):
        print(type(self.x_train))
        print(self.x_train.shape)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.x_val, self.y_val),
            validation_split=0.2,
            verbose=1,
        )
        return self.model

    def save_trained_model(self, checkpoint_path):
        self.model.save(checkpoint_path)
