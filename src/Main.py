from data_processing.bin_img_processing import preprocess

image_data_path = r"C:\Users\azihd\OneDrive\Documents\binary-classification-w-perceptrons\PetImages"

X, y = preprocess(image_data_path)
