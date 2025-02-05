from data_processing.bin_classif_processing_utils import preprocess, image_preprocess
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np

class BinClassifProcessor:

    def __init__(self, dir):
        self.dir = dir
        self.processed_data_path = Path(__file__).parent / "processed_data_dir" / "processed_data.npz"

    def process(self, max_features, standardize=False, images=False):
        # Obtain data
        if images:
            self.X, self.y = image_preprocess(self.dir)
        else:
            self.X, self.y = preprocess(self.dir, "url", "is_spam", max_features, standardize)

        # Split the data into training and testing (90% for training, 10% for testing)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.1,
                                                                                random_state=42)
        # Compress and store data
        np.savez_compressed(self.processed_data_path, X=self.X_train, y=self.y_train, X_test=self.X_test,
                                                                                y_test=self.y_test)
        print(f"Data saved to {self.processed_data_path}")

    def load(self):
        data = np.load(self.processed_data_path)
        self.X_train, self.X_test, self.y_train, self.y_test = data['X'], data['y'], data['X_test'],\
                                                                                data['y_test']
        data.close()
        return self.X_train, self.X_test, self.y_train, self.y_test
