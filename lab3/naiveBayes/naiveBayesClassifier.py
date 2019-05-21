import util
import math

class NaiveBayesClassifier(object):
    """
    See the project description for the specifications of the Naive Bayes classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels, smoothing=0, logTransform=False, featureValues=util.Counter()):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = int(smoothing) # this is the smoothing parameter, ** use it in your train method **
        self.logTransform = logTransform
        self.featureValues = featureValues # empty if there is no smoothing

    def fit(self, trainingData, trainingLabels):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the smoothed estimates so that they can be used to classify.
        
        trainingData is a list of feature dictionaries.  The corresponding
        label lists contain the correct label for each instance.

        To get the list of all possible features or labels, use self.features and self.legalLabels.
        """
        self.features = trainingData[0].keys() # the names of the features in the dataset
        self.prior = util.Counter() # probability over labels
        self.conditionalProb = util.Counter() # Conditional probability of feature feat for a given class having value v

        # ------------COUNTING-------------------
        prior_cnt, cond_prob_cnt = util.Counter(), util.Counter()
        for sample, output in zip(trainingData, trainingLabels):
            prior_cnt[output] += 1
            for feature in self.features: cond_prob_cnt[(feature, output, sample[feature])] += 1

        # ------------SMOOTHING--------------------
        '''
        t = number of times event occurs
        n = total number of trials
        d = number of different classes (OR len(distinct(output)))
        
        Basic: P(Hi) = t/n
        Smoothed: P(Hi) = (t + k)/(n + k*d)
        '''

        # P(Hi) + smoothing
        for output_class in self.legalLabels:
            smoothed = float(prior_cnt[output_class] + self.k)/(len(trainingLabels) + self.k * len(self.legalLabels))
            self.prior[output_class] = smoothed

        # P(Fj | Hi) + smoothing
        for key, count in cond_prob_cnt.iteritems():
            feature, output, value = key
            feature_value = 0 if feature not in self.featureValues else len(self.featureValues[feature])
            smoothed = float(count + self.k)/(prior_cnt[output] + self.k * feature_value)
            self.conditionalProb[key] = smoothed


    def predict(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """

        guesses = []
        self.posteriors = [] # posterior probabilities are stored for later data analysis.
        
        for instance in testData:
            if self.logTransform:
                posterior = self.calculateLogJointProbabilities(instance)
            else:
                posterior = self.calculateJointProbabilities(instance)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses


    def calculateJointProbabilities(self, instance):
        """
        Returns the joint distribution over legal labels and the instance.
        Each probability should be stored in the joint counter, e.g.
        Joint[3] = <Estimate of ( P(Label = 3, instance) )>

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        return self.calculateProbabilities(instance, lambda x: x)


    def calculateLogJointProbabilities(self, instance):
        """
        Returns the log-joint distribution over legal labels and the instance.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, instance) )>

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        return self.calculateProbabilities(instance, lambda x: math.log(x))

    def calculateProbabilities(self, instance, function):

        joint = util.Counter()

        for label in self.legalLabels:
            # calculate the joint probabilities for each class
            # P(Hi)*P(Fj*...*Fk | Hi)
            prob = self.prior[label]

            for feature in self.features:
                prob *= self.conditionalProb[(feature, label, instance[feature])]

            joint[label] = function(prob)

        return joint