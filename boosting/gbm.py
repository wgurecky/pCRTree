#!/usr/bin/python3
##!
# \brief Gradient boosted regression trees
# \date Feb 3 2017
# \author William Gurecky
##
import numpy as np
from scipy.optimize import minimize
from dtree.regress import RegTree

class GBRTmodel(object):
    """!
    @brief Gradient boosted regression tree model.

    Implemented regularization techniques:
    - shrinkage via tunable learning rate
    - stochastic gradient boosting
    """
    def __init__(self, x, y, maxTreeDepth=3, lossFn='se'):
        """!
        @param x (nd_array) Explanatory variable set.  Shape = (N_support_pts, Ndims)
        @param y (1d_array) Response variable vector. Shape = (Ndims,)
        @param maxTreeDepth  Maximum depth of each weak learner in the model.
            Equal to number of possible interactions captured by each tree in the GBRT.
        @param lossFn  Target function to minimize at each iteration of boosting
        """
        self.maxTreeDepth = 3
        self.x = x
        self.y = y
        # init 0th tree in gbm model
        self._trees = [ConstModel(x, y)]
        self._treeWeights = [1.0]
        # value of output regression surface at self.x
        self._F = self._trees[0].predict(self.x)[1]

    def predict(self, testx):
        if len(np.shape(testX)) == 1:
            testX = np.array([testX]).T
        fHat = np.zeros(testX.shape[0])
        for weight, tree in zip(self._treeWeight, self._trees):
            fHat += weight * tree.predict(testX)[1]
        return fHat

    @property
    def F(self):
        return self._F

    def getTrainingSurface(self):
        return self.x, self._F

    @property
    def treeWeights(self):
        return self._treeWeights

    @property
    def trees(self):
        return self._trees

    def train(self, learning_rate=1.0, subset=0.7, maxIterations=5):
        """!
        @brief Train the regression tree model by the gradient boosting
        method.
        Initilize model with constant value:
        \f[
        F_{m=0} = mean(y)
        \f]
        Where y is the input traing data response vector.

        At each iteration fit weak learner (\f[ h_m \f]) to pseudo-residuals:
        \f[
        r(x)_m = -\nabla L(y, \hat y) = \frac{\sum_i{(y_i - f(x_i)})^2}{\partial f(x_i)}
        \f]

        Next, seek to minimize:
        \f[
        \frac{\sum_i( L(y_i, F_m + \gamma_m h(x_i)_m)) }{\partial \gamma_m} = 0
        \f]

        Update the model:
        \f[
        F_{m} = F_{m-1} + \gamma_m F_m
        \f]
        @param learning_rate  Scale the influence of each tree in model:
            Model is updated acording to \f[ F = F_{m-1} + lr (R_m) \f]
            Where \f[R_m \f] are the residuals prediced by the mth tree
        @param subset  Fraction of avalible data used for training in any given
            boosted iteration.
        @@param maxIterations  Max number of boosted iterations.
        """
        print("Iteration | Training Err | Tree weight ")
        print("========================================")
        for i in range(maxIterations):
            # fit learner to pseudo-residuals
            self._trees.append(RegTree(self.x,
                                      -self.nablaLoss(self.F()),
                                      maxDepth=self.maxTreeDepth,
                                      )
                              )
            self._trees[-1].fitTree()
            # define minimization problem
            lossSum = lambda gamma: np.sum(self._seLoss(gamma * self._trees[-1].predict(self.x)[1]))
            gamma_ = minimize(lossSumm, 1.0, method='SLSQP').x[0]
            self._treeWeights.append(gamma_)
            self._F = self._F + gamma_ * learning_rate * self._trees[-1].predict(self.x)[1]
            print(str(i) + " |  " + str(self.trainErr(i)) + " |  " + str(gamma_))

    def trainErr(self, loss='se'):
        """!
        @brief Training error at iteration m:
        \f[
        E_m = \sum_i L(F_{m_i} - y_i)
        \f]
        return (float) training error
        """
        return np.sum(self._seLoss(self._F))

    def nablaLoss(self, yHat, loss='se'):
        """!
        @brief Jacobian of loss function. Computes:
        \f[
        -\nabla L(y, \hat y) = \frac{\sum_i{(y_i - f(x_i)})^2}{\partial f(x_i)}
        \f]
        Where \f[ f() \f] is the current model
        @param yHat  (1d_ndarray) predicted responses
        @param loss (str) Loss function name
        """
        if loss is 'se':
            return (self.y - yHat) * 0.5
        else:
            raise NotImplementedError

    def _seLoss(self, yHat):
        """!
        @brief Squared error loss function.
        @param yHat  (1d_ndarray) predicted responses
        @return 1d_array  \f[ L = (y_i - \hat y_i) **2) \f]
        """
        # No need to divide by N here, outcome of minimization is the same
        return (self.y - yHat) ** 2.0


class ConstModel(object):
    def __init__(self, x, y, *args, **kwargs):
        self._yhat = np.mean(y)

    def predict(self, x, *args, **kwrags):
        return self._yhat * np.ones(len(x))
