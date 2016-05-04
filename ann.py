import util
import math
import operator
import random
import numpy as np
PRINT = True

class NeuralNetworkClassifier:
  """
  Neural Network classifier.

  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations, alpha):
    self.legalLabels = legalLabels
    self.type = "ann"
    self.max_iterations = max_iterations
    self.alpha = alpha

  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    """
    The training loop for the ann passes through the training data several
    times and updates the theta vector and bias vector to minimize the cost.
    """

    self.features = trainingData[0].keys() # could be useful later

    # Set number of units for three layers
    self.s1 = len(self.features)
    self.s2 = 30
    self.s3 = len(self.legalLabels)

    # Initialize bias vector and theta vector for input level and hidden level
    self.bias1 = np.random.randn(self.s2, 1)
    self.bias2 = np.random.randn(self.s3, 1)
    self.theta1 = np.random.randn(self.s2, self.s1)
    self.theta2 = np.random.randn(self.s3, self.s2)

    size = 10 if len(self.legalLabels) == 10 else 45# smaller section size
    # alpha = 3.0  # learning rate
    n = len(trainingData)

    for iteration in xrange(self.max_iterations):
      if iteration % 10 == 0:
        print "Starting iteration %d out of %d ..." % (iteration, self.max_iterations)
      # Seperate one big data set into smaller sets
      for begin in range(0, n, size):

        dev_b1 = np.zeros(self.bias1.shape)
        dev_b2 = np.zeros(self.bias2.shape)
        dev_t1 = np.zeros(self.theta1.shape)
        dev_t2 = np.zeros(self.theta2.shape)

        for i in range(begin, begin + size):
          features = trainingData[i]
          label = trainingLabels[i]
          x = np.reshape(features.values(), (len(self.features), 1))

          y = np.zeros((10, 1)) if len(self.legalLabels) == 10 else np.zeros((2, 1))
          y[label] = 1.0

          # forward propagation
          a1 = x
          z1 = np.dot(self.theta1, a1) + self.bias1
          a2 = sigmoid(z1)
          z2 = np.dot(self.theta2, a2) + self.bias2
          a3 = sigmoid(z2)

          # backward propagation
          delta = (a3 - y) * sigmoidPrime(z2)
          delta_b2  = delta
          delta_t2 = np.dot(delta, a2.transpose())
          delta = np.dot(self.theta2.transpose(), delta) * sigmoidPrime(z1)
          delta_b1 = delta
          delta_t1 = np.dot(delta, a1.transpose())

          dev_b1 = dev_b1 + delta_b1
          dev_b2 = dev_b2 + delta_b2
          dev_t1 = dev_t1 + delta_t1
          dev_t2 = dev_t2 + delta_t2

        self.theta1 = self.theta1 - (self.alpha / size) * dev_t1
        self.theta2 = self.theta2 - (self.alpha / size) * dev_t2
        self.bias1 = self.bias1 - (self.alpha / size) * dev_b1
        self.bias2 = self.bias2 - (self.alpha / size) * dev_b2


  def classify(self, data):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.

    Recall that a datum is a util.counter...
    """
    guesses = []
    for datum in data:
      a1 = np.reshape(datum.values(), (len(datum), 1))
      a2 = sigmoid(np.dot(self.theta1, a1) + self.bias1)
      a3 = sigmoid(np.dot(self.theta2, a2) + self.bias2)
      guesses.append(np.argmax(a3))
    return guesses

def sigmoid(z):
  """The sigmoid function."""
  return 1.0 / (1.0 + np.exp(-z))

def sigmoidPrime(z):
  """Derivative of the sigmoid function."""
  return sigmoid(z) * (1 - sigmoid(z))