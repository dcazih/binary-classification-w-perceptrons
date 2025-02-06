import sys
import os
import numpy as np
from pathlib import Path
from data_processing.BinClassifProcessor import BinClassifProcessor
from data_processing.bin_classif_processing_utils import test_model, process_url, get_valid_url
from src.Perceptron import Perceptron
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

data_path = Path(__file__).parent / r"data\url_spam_classification.csv"
handler = BinClassifProcessor(data_path)

# Get data
getData = False
if getData:
    if handler.processed_data_path.exists():
        print("Loading previous data ... ")
        X, y, X_test, y_test = handler.load()
        print("Done!")
    else:
        print("Processing data ... ")
        vectorizer = handler.process(10000)  # param: max_features my system can handle
        print("Loading data ... ")
        X, y, X_test, y_test = handler.load()
        print("Done!")

# Start data learning
isTrained = True
if not isTrained:
    perceptron = Perceptron()
    trained_model = perceptron.train(X, y)
    trained_model.save_weights("trained_model_data.npz")
    print("Done!")
else:
    data = np.load(Path(__file__).parent / r"trained_model_data.npz")
    trained_model = Perceptron(data["weights"], data["bias"])
    print(f"Weights: {trained_model.w_}\n Bias: {trained_model.b_}")

    #test_model(trained_model, X_test, y_test)
    vectorizer = CountVectorizer(max_features=10000)

    while True:
        url = get_valid_url()
        X = process_url(url, vectorizer)

        print(trained_model.predict(X))



