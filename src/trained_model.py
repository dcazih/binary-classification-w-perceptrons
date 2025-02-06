import sys
import os
import numpy as np
from src.Perceptron import Perceptron
from data_processing.bin_classif_processing_utils import process_url, test_model


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

data = np.load(r"C:\Users\azihd\OneDrive\Documents\binary-classification-w-perceptrons\src\trained_model_data.npz")
trained_perceptron_model = Perceptron(data["weights"], data["bias"])

while True:


    url = input("Check url for spam: ")

    if url


