from data_processing.BinImgProcessor import BinImgProcessor
from data_processing.bin_img_processing_utils import test_model
from src.Perceptron import Perceptron
from matplotlib import pyplot as plt

image_data_path = r"C:\Users\azihd\OneDrive\Documents\binary-classification-w-perceptrons\PetImages"
handler = BinImgProcessor(image_data_path)

# Get data
if handler.processed_data_path.exists():
    print("Loading previous data ... ")
    X, y, X_test, y_test = handler.load()
    print("Done!")
else:
    print("Processing image data ... ")
    handler.process(image_data_path)
    print("Loading data ... ")
    X, y, X_test, y_test = handler.load()
    print("Done!")

# Start data learning
perceptron = Perceptron(eta=0.01, n_epochs=300, random_seed=4078)
trained_model = perceptron.train(X, y)
print("Trained!")

# Plot errors
plt.plot(range(1, len(trained_model.errors_) + 1), trained_model.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

test_model(trained_model, X_test, y_test)
