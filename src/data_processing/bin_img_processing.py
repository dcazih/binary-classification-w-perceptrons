from PIL import Image
from pathlib import Path
import numpy as np

"""
    Functions for processing binary image data for binary classification.

    Functions are designed to:
    - Ensure the correct directory structure and format of image data for binary classification
    - Preprocess the images (flattening and normalization)
    - Generate appropriate labels for the image data (0 or 1)

    The image data is assumed to be organized into two subdirectories, each containing images of one class label.
    """


def format(img_dir):
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

    # Create directory path and its subdirectories
    img_data_path = Path(img_dir)
    subdirs = [d for d in img_data_path.iterdir() if d.is_dir()]  # finds the given path's subdirectories

    # Ensure subdirectory requirement
    if len(subdirs) != 2:
        raise ValueError(f"Error: Expected exactly two subdirectories, but found {len(subdirs)}.")

    accepted_img_types = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}

    # A function to check if a directory contains any images
    def contains_image(directory):
        return any((f.suffix in accepted_img_types) for f in directory.iterdir() if f.is_file())

    # Ensure image subdirectory requirement
    if not all(contains_image(sd) for sd in subdirs):
        raise ValueError("Error: Both subdirectories must contain at least one image file.")

    # Return validated image subdirectories
    print("Format Check: Passed")
    return subdirs[0], subdirs[1]


def preprocess(img_dir):
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

    # Get image data and ensure proper format
    image_data_path = Path(img_dir)
    cat_subdir, dog_subdir = self.format(image_data_path)

    # ==* Process images into numpy vectors to create our training dataset  *==

    cat_img_list, dog_img_list = [], []  # Initialize image arrays

    # Store processed images in numpy array: cat_img_array
    n = 0
    for cat_pic_path in cat_subdir.glob("*.jpg"):

        # Todoo: Remove limit of 20 images when done testing
        if n > 20:
            break
        else:
            n += 1

        cat_img = Image.open(cat_pic_path)
        cat_img.resize((28, 28))  # Resize
        cat_img = cat_img.convert('L')  # Convert to greyscale
        cat_img_array = np.array(cat_img).flatten()  # Flatten image to vector
        cat_img_array = cat_img_array / 255.0  # Normalize pixel values (0, 1)

        cat_img_list.append(cat_img_array)  # Append processed image

    # Store processed images in  in numpy array: dog_img_array
    n = 0
    for dog_pic_path in dog_subdir.glob("*.jpg"):

        # Todoo: Remove limit of 20 images when done testing
        if n > 20:
            break
        else:
            n += 1

        dog_img = Image.open(dog_pic_path)
        dog_img = dog_img.resize((28, 28))  # Resize to 28x28
        dog_img = dog_img.convert('L')  # Convert the image to grayscale
        dog_img_array = np.array(dog_img).flatten()  # Flatten image to vector
        dog_img_array = dog_img_array / 255.0  # Normalize pixel values (0, 1)

        dog_img_list.append(dog_img_array)  # Append processed image

    # Create feature matrix (X): a 2D numpy array of numpy vector images (arrays)
    X = np.array(cat_img_list + dog_img_list, dtype=object)
    print("Feature Matrix: Created Successfully")
    print("flattened cat image: ", X[:1])

    # Create label vector (y): a 1D numpy array, denoting 1 for cats and 0 for dogs
    cats, dogs = np.ones(len(cat_img_list), dtype=int), np.zeros(len(dog_img_list), dtype=int)
    y = np.concatenate((cats, dogs))
    print("Class Labels: Created Successfully")
    print("cat:", y[:1])

    # Randomly permutate X and y
    permutation = np.random.permutation(len(X))
    X, y = X[permutation], y[permutation]

    return X, y

