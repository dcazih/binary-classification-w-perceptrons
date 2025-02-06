from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import numpy as np
import pandas as pd


"""
    Functions for processing binary image data for binary classification.
    
    Functions are designed to:
    - Ensure the correct directory structure and format of image data for binary classification
    - Preprocess the images (flattening and normalization)
    - Generate appropriate labels for the image data (0 or 1)
    
    The image data is assumed to be organized into two subdirectories, each containing images of one class label.
"""
# User test functions
def process_url(url):
    # Convert URL into a dataframe to match preprocessing format
    df = pd.DataFrame([url], columns=["urls"])

    # Transform using the given vectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.transform(df["urls"]).toarray()

    return X

# Standard data functions
def csvFormat(dir):
    data = Path(dir)
    if data.suffix != ".csv":
        raise FileNotFoundError(f"Error: expected file format .cvs but found {data.suffix}")
    return data


def preprocess(dir, feature, label, max_features, standardize):
    data = csvFormat(dir)
    df = pd.read_csv(data)

    vectorizer = CountVectorizer(max_features=max_features)

    # Create feature matrix (X): a 2D numpy array of numpy vector images (arrays)\
    X = vectorizer.fit_transform(df[feature]).toarray()

    # Create label vector (y): a 1D numpy array, denoting 1 for cats and 0 for dogs
    y = np.array(df[label])  # true: url is spam, false: not spam

    # Perform a standardization to sample features (X)
    if standardize:
        X_mean = np.mean(X)
        X_std = np.std(X)
        X = (X - X_mean) / X_std

    return X, y


# Test Functions
def test_model(trained_model, X_test, y_test):
    """
    Test the model with the test data and visualize the results.

    Parameters
    ----------
    trained_model : object
        A trained model that has a `predict` method (your custom model).

    X_test : array-like, shape = [n_samples, n_features]
        Test features.

    y_test : array-like, shape = [n_samples]
        True labels for test data.
    """

    # Make predictions on test data using the trained model
    y_pred = trained_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")

    # Plot misclassified samples
    misclassified_samples = np.where(y_pred != y_test)[0]
    print(f"Misclassified samples: {len(misclassified_samples)} out of {len(y_test)}")

    """
    # display the first 2 misclassified images
    for i in misclassified_samples[:1]:
        plt.imshow(restore_image(X_test[i]), cmap='gray')
        plt.title(f"Pred: {y_pred[i]}, True: {y_test[i]}")
        # plt.show()
    """

    # Confusion Matrix - Shows TP, TN, FP, FN counts
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    # plt.show()

    # Plot the error per sample
    errors = (y_pred != y_test).astype(int)  # 1 if misclassified, 0 if correct
    plt.figure(figsize=(10, 6))
    plt.plot(errors, 'ro', label="Misclassified")
    plt.plot(np.ones_like(errors) * accuracy, 'bo', label="Correct")
    plt.xlabel('Test Samples')
    plt.ylabel('Error')
    plt.title('Misclassified vs Correct Samples')
    plt.legend()
    # plt.show()


# Image Functions
def restore_image(flattened_img_array, image_size=(28, 28)):
    restored_img_array = flattened_img_array.reshape(image_size)  # unflatten to 2D
    restored_img_array = (restored_img_array * 255).astype(np.uint8)  # denormalize
    restored_image = Image.fromarray(restored_img_array, mode='L')  # convert numpy arr to PIL image
    restored_image.show()  # display image
    return restored_image

def imageFormat(img_dir):
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
        The two image subdirectories used for training/testing

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

def image_preprocess(img_dir, augment=True):
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
    cat_subdir, dog_subdir = format(image_data_path)

    # ==* Process images into numpy vectors to create our training dataset  *==

    cat_img_list, dog_img_list = [], []  # Initialize image arrays

    # Store processed images in numpy array: cat_img_array
    n = 0
    for cat_pic_path in cat_subdir.glob("*.jpg"):
        try:
            cat_img = Image.open(cat_pic_path)
            print(cat_img.filename)

            cat_img = cat_img.resize((28, 28), Image.Resampling.LANCZOS)  # Resize
            cat_img = cat_img.convert('L')  # Convert to greyscale

            cat_img_array = np.array(cat_img)
            cat_img_array = cat_img_array.flatten()  # Flatten image to vector
            cat_img_array = cat_img_array / 255.0  # Normalize pixel values (0, 1)

            cat_img_list.append(cat_img_array)  # Append processed image

        except (UnidentifiedImageError, IOError, OSError) as e:
            # If an error occurs (e.g., invalid image), print the error and skip this image
            print(f"Skipping invalid image: {cat_pic_path} - {e}")

    # Store processed images in  in numpy array: dog_img_array
    n = 0
    for dog_pic_path in dog_subdir.glob("*.jpg"):
        try:
            dog_img = Image.open(dog_pic_path)
            print(dog_img.filename)

            dog_img = dog_img.resize((28, 28), Image.Resampling.LANCZOS)  # Resize to 28x28
            dog_img = dog_img.convert('L')  # Convert the image to grayscale

            dog_img_array = np.array(dog_img)
            dog_img_array = dog_img_array.flatten()  # Flatten image to vector
            dog_img_array = dog_img_array / 255.0  # Normalize pixel values (0, 1)

            dog_img_list.append(dog_img_array)  # Append processed image

        except (UnidentifiedImageError, IOError, OSError) as e:
            # If an error occurs (e.g., invalid image), print the error and skip this image
            print(f"Skipping invalid image: {dog_pic_path} - {e}")

    # Create feature matrix (X): a 2D numpy array of numpy vector images (arrays)
    print(len(cat_img_list), len(dog_img_list))
    X = np.array(cat_img_list + dog_img_list, dtype=float)
    print("Feature Matrix: Created")

    # Create label vector (y): a 1D numpy array, denoting 1 for cats and 0 for dogs
    cats, dogs = np.ones(len(cat_img_list), dtype=int), np.zeros(len(dog_img_list), dtype=int)
    y = np.concatenate((cats, dogs))
    print("Class Labels: Created")

    # Randomly permutate X and y
    permutation = np.random.permutation(len(X))
    X, y = X[permutation], y[permutation]

    return X, y
