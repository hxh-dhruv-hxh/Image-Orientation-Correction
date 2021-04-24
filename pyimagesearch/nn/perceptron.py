import numpy as np

class Perceptron:

    def __init__(self, N, alpha=0.1):

        # Inititalizing the weight matrix and storing the learning rate
        self.W = np.random.randn(N+1) / np.sqrt(N)
        self.alpha = alpha

    def step(self, x):

        return 1 if x>0 else 0

    def fit(self, X, y, epochs=10):

        # Inserting bias into the feature matrix
        X = np.c_[X, np.ones(X.shape[0])]

        # Looping over desired no. of epochs
        for epoch in np.arange(0, epochs):

            # Looping over each individual datapoint
            for (x, target) in zip(X, y):

                # Taking the input between the feature matrix and the weight matrix and then passing this into the ...
                # ... step function to get the predictions
                p = self.step(np.dot(X, self.W))

                # Only perform target update if our prediction does not matches the ground truth
                if p != target:
                    # Determine the error
                    error = p - target

                    # Update the weight matrix
                    self.W += -self.alpha * error * x

    def predict(self, X, addBias=True):

        # Ensure that our input is a matrix
        X = np.atleast_2d(X)

        # Check to see if the bias column is to be added
        if addBias:

            X = np.c_[X, np.ones(X.shape[0])]

        return self.step(np.dot(X, self.W))

