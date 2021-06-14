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

data_path = os.path.join(SAKEMAKER_PREFIX, "input", "data")
model_path = os.path.join(SAKEMAKER_PREFIX, "model")
training_path = os.path.join(data_path, "training/training.csv")
validation_path = os.path.join(data_path, "validation/validation.csv")

trainer = Trainer()
trainer.load_data(training_path, validation_path)
trainer.split_data()
trainer.build_model()
trainer.compile_model(DEFAULT_LEARNING_RATE, DEFAULT_DECAY)
trainer.fit_model(DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS)
trainer.save_trained_model(model_path)

