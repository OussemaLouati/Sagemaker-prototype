from keras.models import load_model


class Predictor(object):
    def __init__(self):
        pass

    def load_model(self, model_path):
        self.model = load_model(model_path)
        return self.model

    def predict(self, input):
        return self.model(input)
