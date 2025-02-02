from bin_img_processing_utils import preprocess
from pathlib import Path
import numpy as np

class BinImgProcessor:

    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.processed_data_path = Path(__file__).parent / "processed_data"
        self.X, self.y = preprocess(img_dir)  # process data

        np.savez_compressed("processed_images.npz", X=self.X, y=self.y)  # store processed data
        print(f"Data saved to {self.processed_data_path / 'processed_images.npz'}")

    def reprocess(self, img_dir):
        self.X, self.y = preprocess(img_dir)  # process data
        np.savez_compressed(self.processed_data_path / "processed_images.npz", X=self.X, y=self.y)  # store data
        print(f"Data saved to {self.processed_data_path / 'processed_images.npz'}")




