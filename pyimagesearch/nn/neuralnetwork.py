import numpy as np

class NeuralNetwork:

    def __init__(self, layers, alpha=0.1):

        self.W = []
        self.layers = layers
        self.alpha = alpha

        # start looping from the index of the first layer but stop before we reach the last two layers
        for i in np.arange(0, len(self.layers)-2):

            # Randomly initializing a weight matrix connecting the number of nodes in each respective layers layers
            # ... and adding an extra node for the bias
            w = np.random.randn(layers[i]+1, layers[i+1]+1)
            # Dividing w by square root of number of nodes in the current layer
            self.W.append(w/np.sqrt(layers[i]))

        # The last two layers are special terms where input conncections need a bias term but the output does not
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w/np.sqrt(layers[-2]))

    def __repr__(self):

        # Construct and return a string that represents the network architecture
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))

    def sigmoid(self, x):

        return 1.0/(1 + np.exp(-x))

    # Defining the derivative of sigmoid which will be used in the backward pass
    def sigmoid_deriv(self, x):

        return x * (1 - x)

    def fit(self, X, y, epochs=1000, displayUpdate=100):

        X = np.c_[X, np.ones((X.shape[0]))]

        # Looping over the desired no. of epochs
        for epoch in np.arange(0, epochs):

            # Looping over each and every data point and training our network
            for (x, target) in zip(X, y):

                self.fit_partial(x, target)

            # check to see if we should diplay a training update
            if epoch==0 or (epoch + 1) % displayUpdate == 0:

                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch+1, loss))

    def fit_partial(self, x, y):

        # Constructing our list of output activations for each layer, first activation is a special case
        # ... as its only the feature vectors
        A = [np.atleast_2d(x)]

        # FEEDFORWARD
        # Loop over the layers in the network
        for layer in np.arange(0, len(self.W)):

            net = A[layer].dot(self.W[layer])

            out = self.sigmoid(net)

            A.append(out)

        # BACKPROPOGATION
        error = A[-1] - y

        D = [error * self.sigmoid_deriv(A[-1])]

        for layer in np.arange(len(A)-2, 0, -1):

            delta = D[-1].dot(self.W[layer].T)

            delta = delta * self.sigmoid_deriv(A[layer])

            D.append(delta)

        # Since we have looped our layers in reverse order we have to reverse the deltas
        D = D[::-1]

        # WEIGHT UPDATE PHASE
        # Looping over the layers
        for layer in np.arange(0, len(self.W)):

            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, addbias=True):

        p = np.atleast_2d(X)

        if addbias:

            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in np.arange(0, len(self.W)):

            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p

    def calculate_loss(self, X, targets):

        targets = np.atleast_2d(targets)

        predictions = self.predict(X, addbias=False)

        loss = 0.5 * np.sum((predictions - targets)**2)

        return loss







