import sys 
import os
import joblib
import numpy as np
from pathlib import Path
from data_processing.BinClassifProcessor import BinClassifProcessor
from data_processing.bin_classif_processing_utils import test_model, process_url, get_valid_url
from Perceptron import Perceptron
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

data_path = Path(__file__).parent / r"data\url_spam_classification.csv"
handler = BinClassifProcessor(data_path)

# Get data
getData = True
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

# Start training (False) or testing (True)
isTrained = False
if not isTrained:
    # Train
    perceptron = Perceptron()
    trained_model = perceptron.train(X, y)
    trained_model.save_weights("trained_model_data.npz")
    
    # Plot errors
    plt.plot(range(1, len(trained_model.errors_) + 1), trained_model.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    plt.savefig("good_convergence_50epoch90percent.pdf")  # Save as PDF
    # plt.show()

    print("Done!")
else:
    # Load a perceptron with trained weights and bias
    data = np.load(Path(__file__).parent / r"trained_model_data.npz")
    trained_model = Perceptron(data["w"], data["b"])
    print(f"Weights: {trained_model.w_}\nBias: {trained_model.b_}")

    test_model(trained_model, X_test, y_test)

    # Load vectorizer used in training
    vectorizer = joblib.load(r"C:\Users\azihd\OneDrive\Documents\binary-classification-w-perceptrons\vectorizer.pkl")

    # Prompt user to check a url for spam
    while True:
        url = get_valid_url()
        X = process_url(url, vectorizer)

        if trained_model.predict(X) == [0]:
            print("URL is not spam (most likely)\n")
        else:
            print("URL is spam\n")


