import numpy as np
class Perceptron:
    """
    Perceptron Classifiers

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

    def __init__(self, eta=0.01, n_epochs=50, random_seed=4078):
        self.eta = eta
        self.n_epochs = n_epochs
        self.random_seed = random_seed

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


