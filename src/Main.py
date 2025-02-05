from data_processing.BinClassifProcessor import BinClassifProcessor
from data_processing.bin_classif_processing_utils import test_model
from src.Perceptron import Perceptron
from matplotlib import pyplot as plt

data_path = r"C:\Users\azihd\OneDrive\Documents\binary-classification-w-perceptrons\data\url_spam_classification.csv"
handler = BinClassifProcessor(data_path)

# Get data
if handler.processed_data_path.exists():
    print("Loading previous data ... ")
    X, y, X_test, y_test = handler.load()
    print("Done!")
else:
    print("Processing data ... ")
    handler.process(4000)  # param: max_features my system can handle
    print("Loading data ... ")
    X, y, X_test, y_test = handler.load()
    print("Done!")

# Start data learning
perceptron = Perceptron()
trained_model = perceptron.train(X, y,)
print("Done!")

# Plot errors
plt.plot(range(1, len(trained_model.errors_) + 1), trained_model.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

test_model(trained_model, X_test, y_test)
