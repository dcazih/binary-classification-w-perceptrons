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

        # Initialize a random normal dist of weights
        rgen = np.random.RandomState(self.random_seed)

        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])

        # Initialize bias and errors
        self.b_ = np.float_(0.0)
        self.errors_ = []

        # Run training
        for i in range(self.n_epochs):
            errors = 0
            for xi, target in zip (X, y):
                update = self.eta * (target - self.predict(xi))  # update = η * (yi − ̂yi)
                self.w_ += update * xi  # w_ = w_ + update * xi
                self.b_ += update  # b_ = b_ + update
                errors += int(update != 0.0)  # Check prediction: 1 wrong, 0 correct
            self.errors_.append(errors)
        return self
