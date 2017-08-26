#!/usr/bin/python3
##!
# \brief Boosted classification trees using the SAMME method
# \date Feb 10 2017
# \author William Gurecky
##
from __future__ import division
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
    def __init__(self, max_depth=3, learning_rate=1.0, subsample=1.0, loss='exp', **kwargs):
        """!
        @param max_depth  Maximum depth of each weak learner in the model.
            Equal to number of possible interactions captured by each tree in the GBCT.
        @param learning_rate  Scale the influence of each tree in model.
        @param subsample  Fraction of avalible data used for training in any given
            boosted iteration.
        @param loss  (str) Target function to minimize at each iteration of boosting
            string in ("exp")
        """
        self.max_depth = max_depth
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
        self._K = None

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
        len_x = testX.shape[0]
        unique_cls_labels = np.sort(np.unique(self.y))
        histCols = np.zeros((len(unique_cls_labels), len_x))
        for i, tree in enumerate(self._trees):
            tree_predictions = tree.predict(testX)
            for v, k in enumerate(unique_cls_labels):
                # where did this tree correcly predict?
                Ic = np.zeros(len_x)
                corrMask = (k == tree_predictions)
                Ic[corrMask] = 1
                histCols[v] += self._treeWeights[i] * Ic
        hist = histCols.T
        return hist

    def predictClassProbs(self, testX, **kwargs):
        """!
        @brief Compute class probability distribution at testX
        @param testX (nd_array) evaluate model at these points
        @return classProbs shape=(Nclasses, len(testX))
        """
        hist = self._buildHist(testX)
        # for col in hist
        sumC_K = np.sum( np.exp((1 / (self._K - 1.)) * hist.T), axis=0)
        classProbs = []
        for col in hist.T:
            probC_K = np.exp((1 / (self._K - 1.)) * col)
            classProbs.append(probC_K / sumC_K)
        return np.array(classProbs).T

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

    def train(self, x, y, n_estimators=5, warmStart=0, **kwargs):
        """!
        @brief Train the classification tree model by the SAMME method.
        @param x (nd_array) Explanatory variable set.  Shape = (N_support_pts, Ndims)
        @param y (1d_array) Response class vector. Shape = (N_support_pts,)
        @param n_estimators  Max number of boosted iterations.
        """
        print("Iteration | Training Err | Tree weight ")
        print("========================================")
        xTest = kwargs.pop("xTest", None)
        yTest = kwargs.pop("yTest", None)
        status = []
        self.x = x
        self.y = y
        lenY = len(self.y)
        y_weights = np.ones(lenY) / lenY
        self._K = len(np.unique(self.y))
        for i in range(n_estimators):
            # subsample training data
            sub_idx = np.random.choice([True, False], len(y), p=[self.subsample, 1. - self.subsample])
            # Fit classification tree to training data with current weights
            self._trees.append(ClsTree(self.x[sub_idx], self.y[sub_idx],
                maxDepth=self.max_depth, weights=y_weights[sub_idx]))
            # y_weights = self._trees[i].weights
            self._trees[i].fitTree()
            #
            # where did this tree incorrectly predict?
            Ic = np.zeros(lenY)
            corrMask = (self.y != self._trees[i].predict(self.x))
            Ic[corrMask] = 1
            #
            # Compute weighted error of current classification tree
            err = np.sum(y_weights * Ic) / np.sum(y_weights)
            #
            # Compute current tree weight
            self._treeWeights.append(self.learning_rate *
                (np.log((1. - err) / err) + np.log(self._K - 1)))
            #
            # Compute new data weights: up-weight were we were wrong
            y_weights *= np.exp(self._treeWeights[i] * Ic)
            y_weights /= np.sum(y_weights)
            #
            # Store status
            if i % 10 == 0:
                fraction_error = self.fracError(x, y)
            else:
                fraction_error = np.nan
            status.append([i, fraction_error, err])
            # status.append([i, self.fracError(x, y), err])
            print(" %4d     | %4e | %.3f " % (status[i][0], status[i][1], status[i][2]))
        return np.array(status)

    def fracError(self, x, y):
        """!
        @brief Compute fraction of misclassified data points
        @param x (nd_array) Explanatory variable set.  Shape = (N_support_pts, Ndims)
        @param y (1d_array) Response class vector. Shape = (N_support_pts,)
        @return fraction of misclassified points
        """
        yhat = self.predict(x)
        corrMask = (yhat != y)
        return np.sum(corrMask) / len(y)

    @property
    def feature_importances_(self, normed=True):
        total_sum = np.zeros(np.shape(self.x)[1])
        for weight, tree in zip(self._treeWeights, self._trees):
            tree_importance = tree.feature_importances_()
            total_sum += tree_importance * 1.0
        print("*** TOTAL SUM IMPORTANCES ***")
        print(total_sum)
        importances = total_sum / np.sum(self._treeWeights)
        return importances / np.sum(importances)


if __name__ == "__main__":
    pass
