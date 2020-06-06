import numpy as np

class MLPTwoLayers:

    # DO NOT adjust the constructor params
    def __init__(self, input_size=3072, hidden_size=100, output_size=10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.input = None

        self.w1 = np.random.normal(0, 1, (input_size, self.hidden_size))
        self.b1 = np.random.normal(0, 1, (1, self.hidden_size))
        self.z1 = None
        self.a1 = None

        self.w2 = np.random.normal(0, 1, (self.hidden_size, output_size))
        self.b2 = np.random.normal(0, 1, (1, self.output_size))
        self.z2 = None
        self.a2 = None

        self.delta2 = None
        self.dE_dw2 = None
        self.delta1 = None
        self.de_dw1 = None

        self.alpha = 0.01

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def loss(self, predictions, label):
        self.label = label
        return np.sum(np.subtract(predictions, label)**2 / predictions.shape[0])

    def d_loss(self):
        return np.subtract(self.a2, self.label)

    def forward(self, features):
        """
            Takes in the features
            returns the prediction
        """
        self.input = features
        self.z1 = np.dot(np.transpose( ), self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def backward(self):
        """
            Adjusts the internal weights/biases
        """
        self.delta2 = self.d_sigmoid(self.z2) * self.d_loss()
        self.delta1 = self.d_sigmoid(self.z1) * np.dot(self.delta2, np.transpose(self.w2))

        self.dE_dw2 = np.dot(np.transpose(self.a1), self.delta2)
        self.dE_dw1 = np.dot(self.input.reshape(-1, 1), self.delta1)

        self.w2 -= self.alpha * self.dE_dw2
        self.w1 -= self.alpha * self.dE_dw1

        self.b2 -= self.alpha * self.delta2
        self.b1 -= self.alpha * self.delta1
