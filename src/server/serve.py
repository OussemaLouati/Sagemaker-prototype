import os
import sys

import numpy as np
from flask import Flask, jsonify, request

from components.Predictor import Predictor
from components.Trainer import Trainer

# CONSTANTS
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_DECAY = 1e-4
DEFAULT_BATCH_SIZE = 256
SAKEMAKER_PREFIX = "/opt/ml"

# SELECT TRAINING OR SERVING ROUTINES


model_path = os.path.join(SAKEMAKER_PREFIX, "model")

prediction_server = Flask(__name__)
predictor = Predictor()
predictor.load_model(model_path)

@prediction_server.route("/predict", methods=["POST"])
def process_prediction():
    """Receives a prediction request and return the forward pass results."""
    try:
        input = np.array(request.json, dtype=np.float32)
        prediction = predictor.predict(input).numpy().tolist()
        return jsonify(result=prediction, status="Prediction succeeded")
    except Exception as err:
        return jsonify(result=None, status=f"Prediction failed: {err}")

prediction_server.run()
