from data_processing.bin_classif_processing_utils import test_model
from pathlib import Path
import numpy as np

class Perceptron:
    """Perceptron Classifiers

    Parameters
    __________
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_epochs : int
        Num of passes over the training dataset
    random_seed : int
        Random number generator seed for random weight initialization

    Attributes
    __________

    w_ : 1d_array
        Final weights after training
    b_ : Scalar
        Final bias after training
    errors_ : list
        Number of classifications in each epoch

    """

    def __init__(self, w_=None, b_=None, eta=0.00005, n_epochs=250, random_seed=76, permutate=True):
        self.permutate = permutate

        self.eta = eta
        self.n_epochs = n_epochs
        self.random_seed = random_seed

        self.w_ = w_
        self.b_ = b_


    def train(self, X, y):
        """Fit training data

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of examples
            and n_features is the number of features.

        y : array-like, shape = [n_examples]
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself."""

        # Initialize weights, bias and errors
        rgen = np.random.RandomState(self.random_seed)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])  # a random normal dist of weights
        self.b_ = np.float64(0.0)
        self.errors_ = []

        try:

            # Run training
            print("Training Model ...")
            for epoch in range(self.n_epochs):

                if self.permutate:
                    permutation = np.random.permutation(len(X))
                    X, y = X[permutation], y[permutation]

                print(f"Epoch {epoch + 1}/{self.n_epochs}")
                errors = 0
                for xi, target in zip (X, y):
                    update = self.eta * (target - self.predict(xi))  # update = η * (yi − ̂yi)
                    self.w_ = self.w_ + update * xi  # w_ = w_ + update * xi
                    self.b_ += update  # b_ = b_ + update
                    errors += int(update != 0.0)  # Check prediction: 1 wrong, 0 correct
                self.errors_.append(errors)
        except Exception:
            test_model(self, X, y)

        return self

    def raw_score(self, X):
        """ Calculate pre activation output"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """
        Applies our desired activation function: step
        And returns a class label
        """
        return np.where(self.raw_score(X) >= 0.0, 1, 0)

    def save_weights(self, dest):
        np.savez(Path(__file__).parent / dest, w=self.w_, b=self.b_, e=self.errors_)
