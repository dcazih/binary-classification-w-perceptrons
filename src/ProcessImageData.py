from PIL import Image
from pathlib import Path
import numpy as np

class ProcessBinaryImgData:
    """
    A class for processing binary image data for binary classification.

    This class is designed to:
    - Ensure the correct directory structure and format of image data for binary classification
    - Preprocess the images (flattening and normalization)
    - Generate appropriate labels for the image data (0 or 1)

    The image data is assumed to be organized into two subdirectories, each containing images of one class.
    """

    def __init__(self):
        None

    def image_data_format_check(self, img_dir):
        """
        Function ensures given path img_dir contains the required formatting:
            - Exactly two folders
            - Each folder must contain at least one image file
            - Otherwise error is thrown

        Parameters
        __________
        img_dir : string
            A path to directory containing image data

        Return
        ______
        (subDir1, subDir2) : tuple
            The two image subdirectories used for training

        """


    def preprocessing(self, img_dir):
        """
        Process the image data to flatten and normalize each image, label them
        and return them as X and y, our desired calculation parameters

        Parameters
        __________
        img_dir : string
            A path to directory containing image data

        Return
        ______
        (X, y) : tuple
            X - 2D numpy array where each elem is an image flattened into a 1D numpy array
            y - Class labels for images where 1 - first image folder (cat), 0 - second image folder(dog)
            Both arrays randomly permutated together
        """




