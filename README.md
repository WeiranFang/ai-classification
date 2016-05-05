# ai-classification
Hand written digits recognition and face recognition using three classifiers: Naive Bayes, Perceptron, Neural Network.

Overview
--------
Optical character recognition (OCR) is the task of extracting text from image sources. Our
target of this project is to implement good algorithms for two tasks: handwriting digit recognition
and human face recognition. In our work, we implemented three different learning
algorithms: Naive Bayes, Perceptron, Neural Network, and three different feature extraction
methods: Basic Features, Concavity Features, Sobel Edge Features. We tested all the three
algorithms on all types of features and made comparison on both accuracy and efficiency. In
our experiments, the best outcome is 91% on accuracy when using Neural Network on Sobel
Edge features with learning rate of 3.0, iteration number of 60, and training data size of 5000.

Dataset
-------
The dataset used for training, validation, and testing contains two parts: 
the DIGIT data and the FACE data. The data size of these two sets is shown as follows:

              Training  Validation  Test
    DIGITS        5000        1000   500
    FACES          451         301    50

Usage
-----
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f basic -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the basicFeatureExtractorDigits function to get the features
                  on the digits dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data of default size 100.

Options
-------
  -h, --help            show this help message and exit
  -c CLASSIFIER, --classifier=CLASSIFIER
                        The type of classifier [Default: mostFrequent]
  -d DATA, --data=DATA  Dataset to use [Default: digits]
  -t TRAINING, --training=TRAINING
                        The size of the training set [Default: 100]
  -f FEATURES, --features=FEATURES
                        The type of features [Default: basic]
  -w, --weights         Whether to print weights [Default: False]
  -k SMOOTHING, --smoothing=SMOOTHING
                        Smoothing parameter (ignored when using --autotune)
                        [Default: 2.0]
  -a, --autotune        Whether to automatically tune hyperparameters
                        [Default: False]
  -i ITERATIONS, --iterations=ITERATIONS
                        Maximum iterations to run training [Default: 3]
  -s TEST, --test=TEST  Amount of test data to use [Default: 100]
  -p ALPHA, --alpha=ALPHA
                        Learning rate for neural network [Default: 1.0]
