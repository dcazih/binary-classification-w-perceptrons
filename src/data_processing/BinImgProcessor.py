from data_processing.bin_img_processing_utils import preprocess
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np

class BinImgProcessor:

    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.processed_data_path = Path(__file__).parent / "processed_data" / "processed_images.npz"

    def process(self, img_dir):
        # Obtain image data
        self.X, self.y = preprocess(img_dir)

        # Split the data into training and testing (90% for training, 10% for testing)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.1,
                                                                                random_state=42)
        # Compress and store data
        np.savez_compressed(self.processed_data_path, X=self.X_train, y=self.y_train, X_test=self.X_test,
                                                                                y_test=self.y_test)
        print(f"Data saved to {self.processed_data_path}")

    def load(self):
        image_data = np.load(self.processed_data_path)
        self.X_train, self.X_test, self.y_train, self.y_test = image_data['X'], image_data['y'], image_data['X_test'],\
                                                                                image_data['y_test']
        image_data.close()
        return self.X_train, self.X_test, self.y_train, self.y_test
