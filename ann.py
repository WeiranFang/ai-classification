import util
import math
import random
import numpy
PRINT = True

class NeuralNetworkClassifier:
  """
  Neural Network classifier.

  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "ann"
    self.max_iterations = max_iterations

    self.alpha = 3.0 # learning rate
    self.lmbda = 0.1
    # self.randomRange = 0.5
    self.theta1 = util.Counter() # Weights for the input layer
    self.theta2 = util.Counter() # Weights for the hidden layer
    self.a1 = util.Counter()
    self.a2 = util.Counter()
    self.a3 = util.Counter()
    self.y = util.Counter()


  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    """
    The training loop for the perceptron passes through the training data several
    times and updates the weight vector for each label based on classification errors.
    See the project description for details.

    Use the provided self.weights[label] data structure so that
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    (and thus represents a vector a values).
    """

    self.features = trainingData[0].keys() # could be useful later
    # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
    # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

    # Set number of units for three layers
    self.s1 = len(self.features)
    self.s2 = len(self.features)
    self.s3 = len(self.legalLabels)

    # Use gaussian distribution to initialize weights for theta1 and theta2
    for i in range(1, self.s2 + 1):
      for j in range(self.s1 + 1):
        self.theta1[(i, j)] = random.gauss(0,1)
    for i in range(1, self.s3 + 1):
      for j in range(self.s2 + 1):
        self.theta2[(i, j)] = random.gauss(0,1)

    bigDelta1 = util.Counter()
    bigDelta2 = util.Counter()
    for iteration in range(self.max_iterations):
      print "Starting iteration ", iteration, "..."
      size = len(trainingData) / 10
      for begin in range(0, len(trainingData), size):
        for i in range(begin, begin + size):
          features = trainingData[i]
          label = trainingLabels[i]
          self.setY(label)

          # Use forward propagation to get the output layer units
          self.setA1(features)
          self.setA2()
          self.setA3()  # a3 is the output layer units

          # print "a1: ", self.a1
          # print "a3: ", self.a3
          # print "y: ", self.y
          # print "label: ", label

          # Use backward propagation to calculate delta for each units
          delta3 = self.a3 - self.y
          delta2 = util.Counter()
          for j in range(1, self.s3 + 1):
            for k in range(self.s2 + 1):
              delta2[k] += self.theta2[(j, k)] * delta3[j]

          # Calculate big delta for layer one
          for j in range(1, self.s2 + 1):
            for k in range(self.s1 + 1):
              bigDelta1[(j, k)] += self.a1[k] * delta2[j]

          # Calculate big delta for layer two
          for j in range(1, self.s3 + 1):
            for k in range(self.s2 + 1):
              bigDelta2[(j, k)] += self.a2[k] * delta3[j]

        # Calculate the partial derivatives of the cost function corresponding to each theta,
        # and then update theta to minimize the cost
        D1 = util.Counter()
        D2 = util.Counter()
        for i in range(1, self.s2 + 1):
          for j in range(self.s1 + 1):
            if j == 0:
              D1[(i, j)] = bigDelta1[(i, j)] / len(trainingData)
            else:
              D1[(i, j)] = bigDelta1[(i, j)] / len(trainingData) + self.lmbda * self.theta1[(i, j)]
            self.theta1[(i, j)] -= self.alpha * D1[(i, j)]
        for i in range(1, self.s3 + 1):
          for j in range(self.s2 + 1):
            if j == 0:
              D2[(i, j)] = bigDelta2[(i, j)] / len(trainingData)
            else:
              D2[(i, j)] = bigDelta2[(i, j)] / len(trainingData) + self.lmbda * self.theta2[(i, j)]
            self.theta2[(i, j)] -= self.alpha * D2[(i, j)]


  def setY(self, label):
    """
    Convert label to vector.
    e.g. If label is 6, then the vector should be [0,0,0,0,0,0,1,0,0,0]
    """
    for i in range(len(self.legalLabels)):
      self.y[i + 1] = 0
    self.y[label + 1] = 1

  def setA1(self, features):
    self.a1[0] = 1 # Add bias
    values = features.values()
    for i in range(1, self.s1 + 1):
      self.a1[i] = values[i - 1]

  def setA2(self):
    self.a2[0] = 1 # Add bias
    for j in range(1, self.s2 + 1):
      z = 0
      for i in range(len(self.a1)):
        z += self.theta1[(j, i)] * self.a1[i]
      self.a2[j] = sigmoid(z)

  def setA3(self):
    for j in range(1, self.s3 + 1):
      z= 0
      for i in range(len(self.a2)):
        z += self.theta2[(j, i)] * self.a2[i]
      self.a3[j] = sigmoid(z)

  def classify(self, data):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.

    Recall that a datum is a util.counter...
    """
    guesses = []
    for datum in data:
      self.setA1(datum)
      self.setA2()
      self.setA3()
      guesses.append(self.a3.argMax() - 1)
    return guesses

def sigmoid(z):
  """
  Helper function: calculate the sigmoid of z
  """
  return 1.0 / (1.0 + numpy.exp(-z))
