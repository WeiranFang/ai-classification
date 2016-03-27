# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    "*** YOUR CODE HERE ***"
    # counter = util.Counter()
    #
    # for feature in self.features:
    #   for label in self.legalLabels:
    #     for fi in [1, 0]:
    #       counter[(feature, fi, label)] = 0
    #
    # for i in range(len(trainingData)):
    #   pixels = trainingData[i]
    #   label = trainingLabels[i]
    #   for pixel in pixels.keys():
    #     counter[(pixel, pixels[pixel], label)] += 1
    #
    # conditionalProbabilities = {}
    # for k in kgrid:
    #   for key in counter.keys():
    #     conditionalProbabilities[(key, k)]

    counterDict = {}
    prior = util.Counter()
    for label in self.legalLabels:
      # prior[label] = 0
      for feature in self.features:
        tempCounter = util.Counter()
        for fi in [0, 1]:
          tempCounter[fi] = 0
        counterDict[(feature, label)] = tempCounter

    for i in range(len(trainingData)):
      featureValues = trainingData[i]
      label = trainingLabels[i]
      prior[label] += 1
      for feature in featureValues.keys():
        counterDict[(feature, label)][featureValues[feature]] += 1

    self.prior = util.normalize(prior)

    bestCorrectCount = 0
    bestCondProb = {}
    for k in kgrid:
      # compute conditional probabilities for each feature and label, unker k
      conditionalProbabilities = {}
      for label in self.legalLabels:
        for feature in self.features:
          counterCopy = counterDict[feature, label]
          counterCopy.incrementAll(counterCopy.keys(), k)
          conditionalProbabilities[feature, label] = util.normalize(counterCopy)

      # count the accuracy and mark the minimum
      self.condProb = conditionalProbabilities
      guesses = self.classify(validationData)
      correctCount = 0
      for i in range(len(validationLabels)):
        if (guesses[i] == validationLabels[i]): correctCount += 1
      if correctCount > bestCorrectCount:
        bestCondProb = conditionalProbabilities

    self.condProb = bestCondProb

    # util.raiseNotDefined()
        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses

  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()
    
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    for label in self.legalLabels:
      logJoint[label] = math.log(self.prior[label])
      for key, value in datum.items():
        logJoint[label] += math.log(self.condProb[key, label][value])

    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
       
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresOdds
    

    
      
