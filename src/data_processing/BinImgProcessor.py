from data_processing.bin_img_processing_utils import preprocess
from pathlib import Path
import numpy as np

class BinImgProcessor:

    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.processed_data_path = Path(__file__).parent / "processed_data" / "processed_images.npz"

    def process(self, img_dir):
        self.X, self.y = preprocess(img_dir)  # process data
        np.savez_compressed(self.processed_data_path, X=self.X, y=self.y)  # store data
        print(f"Data saved to {self.processed_data_path}")

    def load(self):
        image_data = np.load(self.processed_data_path)
        X, y = image_data['X'], image_data['y']
        image_data.close()
        return X, y
