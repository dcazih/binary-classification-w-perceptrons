import numpy as np
from data_processing.BinImgProcessor import BinImgProcessor
from src.Perceptron import Perceptron

image_data_path = r"C:\Users\azihd\OneDrive\Documents\binary-classification-w-perceptrons\PetImages"
handler = BinImgProcessor(image_data_path)

if handler.processed_data_path.exists():
    print("Loading previous data ... ")
    X, y = handler.load()
    print("Done!\n")
else:
    print("Processing image data ... ")
    handler.process(image_data_path)
    print("Loading data ... ")
    X, y = handler.load()
    print("Done!\n")





