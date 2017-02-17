#!/usr/bin/python3
##!
# \brief Boosted classification trees using the SAMME method
# \date Feb 10 2017
# \author William Gurecky
##
import numpy as np
from dtree.classify import ClsTree


class GBCTmodel(object):
    """!
    @brief Gradient boosted classification tree model.

    Implemented regularization techniques:
    - shrinkage via tunable learning rate
    - stochastic gradient boosting

    Implemented loss functions:
    - exp (multi-class exp loss used in SAMME method)
    """
    def __init__(self, maxTreeDepth=3, learning_rate=1.0, subsample=1.0, lossFn='exp'):
        """!
        @param maxTreeDepth  Maximum depth of each weak learner in the model.
            Equal to number of possible interactions captured by each tree in the GBRT.
        @param learning_rate  Scale the influence of each tree in model.
        @param subsample  Fraction of avalible data used for training in any given
            boosted iteration.
        @param lossFn  (str) Target function to minimize at each iteration of boosting
            string in ("se", "huber")
        """
        self.maxTreeDepth = maxTreeDepth
        self.learning_rate = learning_rate
        self.subsample = subsample

        # internal storage (write out to external file on request)
        self._trees = []
        self._treeWeights = []
        self._F = None

        # training data
        self.x = np.array([[]])
        self.y = np.array([])

        # number of class labels
        self._K = len(np.unique(self.y))

    def predict(self, testX, **kwargs):
        """!
        @brief Evaluate boosted classification tree model.
        @param testX (nd_array) evaluate model at these points
        @return fHat (1d_array) predicted class labels at testX inputs
        """
        # Build histogram
        hist = self._buildHist(testX)
        # find max likelihood class labels from histogram(s)
        fHat = np.argmax(hist, axis=1)
        return fHat

    def _buildHist(self, testX):
        """!
        @brief Computes the class histogram at input testX locs
        """
        # tree_weight * (weights * (T_i(x) == K))
        histCols = []
        for k in np.sort(np.unique(self.y)):
            counts = np.zeros(len(self.y))
            for i, tree in enumerate(self._trees):
                # where did this tree correcly predict?
                corrPredict = np.zeros((len(self.y), len(self.y)))
                corrMask = (k == self.tree.predict(testX))
                # set diagonal to correctMask
                Ic = np.diag(corrPredict)
                Ic[corrMask] = 1
                corrPredict[np.diag_indices_from(corrPredict)] = Ic
                counts += np.dot(self._treeWeights[i], corrPredict)
            histCols.append(counts)
        return np.array(histCols).T

    def predictClassProbs(self, testX, **kwargs):
        """!
        @brief Compute class probability distribution at testX
        @return classProbs shape=(Nclasses, len(testX)
        """
        hist = self._buildHist(testX)
        # for col in hist
        sumC_K = np.sum( np.exp((1 / (self._K - 1.)) * hist.T), axis=0)
        classProbs = []
        for col in hist.T:
            probC_K = np.exp((1 / (self._K - 1.)) * col)
            classProbs.append(probC_K / sumC_K)
        return np.array(classProbs)

    @property
    def F(self):
        """!
        @brief 1d_array Stored response surface vector
        """
        return self._F

    @property
    def K(self):
        """!
        @brief Integer number of class labels
        """
        return self._K

    @property
    def treeWeights(self):
        return self._treeWeights

    @property
    def trees(self):
        return self._trees

    def train(self, x, y, maxIterations=5, warmStart=0, **kwargs):
        """!
        @brief Train the classification tree model by the SAMME method.
        @param x (nd_array) Explanatory variable set.  Shape = (N_support_pts, Ndims)
        @param y (1d_array) Response class vector. Shape = (N_support_pts,)
        @param maxIterations  Max number of boosted iterations.
        """
        xTest = kwargs.pop("xTest", None)
        yTest = kwargs.pop("yTest", None)
        status = []
        self.x = x
        self.y = y
        y_weights = None
        for i in range(maxIterations):
            # Fit classification tree to training data with current weights
            self._trees.append(ClsTree(self.x, self.y, maxDepth=self.maxTreeDepth, weights=y_weights))
            y_weights = self._trees[i].weights
            #
            # where did this tree incorrectly predict?
            corrPredict = np.zeros((len(self.y), len(self.y)))
            corrMask = (self.y != self._trees[i].predict(self.x))
            Ic = np.diag(corrPredict)
            Ic[corrMask] = 1
            corrPredict[np.diag_indices_from(corrPredict)] = Ic
            # Ic == sparse matrix with 1's on diag where current tree model != training y_i
            #
            # Compute weighted error of current classification tree
            err = np.sum(np.dot(y_weights, corrPredict)) / np.sum(y_weights)
            #
            # Compute current tree weight
            self._treeWeights.append(np.log((1. - err) / err) + np.log(self._K - 1))
            #
            # Compute new data weights
            y_weights = y_weights * np.exp(np.diag(self._treeWeights[i] * corrPredict))
            y_weights /= np.sum(y_weights)


    def trainErr(self):
        """!
        @brief Training error.
        """
        if self._F is not None:
            return self._L.var(self.y, self._F)
        else:
            return None

    def testErr(self, xTest, yTest):
        """!
        @brief Testing error.
        """
        if self._F is not None:
            return self._L.var(yTest, self.predict(xTest))
        else:
            return None
