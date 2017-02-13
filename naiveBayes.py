import util
import classificationMethod
import math
import copy

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
      
    self.features = trainingData[0].keys() # this could be useful for your code later...
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

    # calculating the priors i.e. probability of a label given a training set
    self.prior_prob = []
    for i in self.legalLabels:
      self.prior_prob.append(trainingLabels.count(i)*1.0/len(trainingLabels))
  
    num_features = len(self.features)
    #count = np.zeros((len(self.legalLabels), num_features))
    count = [[0]*num_features for i in xrange(len(self.legalLabels))]

    #counting the occurrences of a feature having same label in training set 
    for i in xrange(len(trainingData)):
      #check redundancy
      d = dict(trainingData[i])
      for j in xrange(num_features):
        if d[self.features[j]] == 1:
          count[trainingLabels[i]][j] += 1
          #print trainingLabels[i], j, count[trainingLabels[i]][j]
    #print count[2]

    self.cond_prob = [[0]*num_features for i in xrange(len(self.legalLabels))]


    for k in kgrid:
      for i in xrange(len(self.legalLabels)):
        for j in xrange(num_features):
          #smoothened probability
          num = (count[i][j] + k) * 1.0
          denom = 0.0
          for l in xrange(len(self.legalLabels)):
            denom += count[l][j] + k
          x_given_y = num/denom
          self.cond_prob[i][j] = x_given_y
          #print i, j
    #print self.cond_prob[2]
    


  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
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
    """
    logJoint = util.Counter()
    
    for y in self.legalLabels:
      prob_y_given_x = math.log(self.prior_prob[y])
      for i in xrange(len(self.features)):
         
        if datum[self.features[i]] == 1:
          prob_y_given_x += math.log(self.cond_prob[y][i])
        else:
          prob_y_given_x += math.log(1 - self.cond_prob[y][i])
      logJoint[y] = prob_y_given_x

    return logJoint


    
      
