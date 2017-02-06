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
    def __init__(self, maxTreeDepth=3, learning_rate=1.0, subsample=1.0, lossFn='se'):
        """!
        @param maxTreeDepth  Maximum depth of each weak learner in the model.
            Equal to number of possible interactions captured by each tree in the GBRT.
        @param learning_rate  Scale the influence of each tree in model:
            Model is updated acording to \f[ F = F_{m-1} + lr (R_m) \f]
            Where \f[R_m \f] are the residuals prediced by the mth tree
        @param subsample  Fraction of avalible data used for training in any given
            boosted iteration.
        @param lossFn  Target function to minimize at each iteration of boosting
        """
        self.maxTreeDepth = 3
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.lossFn = lossFn
        #
        self._trees = [None]
        self._treeWeights = [1.0]
        self._F = np.array([])

    def predict(self, testX):
        """!
        @brief Evaluate gradient boosted regression tree model.
        @param testX (nd_array) evaluate model at these points
        """
        if len(np.shape(testX)) == 1:
            testX = np.array([testX]).T
        fHat = np.zeros(testX.shape[0])
        for weight, tree in zip(self._treeWeights, self._trees):
            fHat += weight * tree.predict(testX)
        return fHat

    @property
    def F(self):
        """!
        @brief 1d_array Stored response surface vector
        """
        return self._F

    def getTrainingSurface(self):
        return self.x, self._F

    @property
    def treeWeights(self):
        return self._treeWeights

    @property
    def trees(self):
        return self._trees

    def train(self, x, y, maxIterations=5, warmStart=0, **kwargs):
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
        @param x (nd_array) Explanatory variable set.  Shape = (N_support_pts, Ndims)
        @param y (1d_array) Response variable vector. Shape = (Ndims,)
        @param maxIterations  Max number of boosted iterations.
        """
        xTest = kwargs.get("xTest", None)
        yTest = kwargs.get("yTest", None)
        if xTest is not None and yTest is not None:
            print("Iteration | Training Err | Testing Err  | Tree weight ")
            print("======================================================")
        else:
            print("Iteration | Training Err | Tree weight ")
            print("========================================")
        # Reset model
        self.x = x
        self.y = y
        self._trees = [ConstModel(x, y)]
        self._treeWeights = [1.0]
        self._F = self._trees[0].predict(self.x)
        for i in range(maxIterations):
            # def sub sample training set
            sub_idx = np.random.choice([True, False], len(y), p=[self.subsample, 1. - self.subsample])
            # fit learner to pseudo-residuals
            self._trees.append(RegTree(self.x[sub_idx],
                                       self.nablaLoss(self.y[sub_idx], self._F[sub_idx]),
                                       maxDepth=self.maxTreeDepth,
                                       )
                              )
            self._trees[-1].fitTree()
            # define minimization problem
            pre_pred_loss = self.trees[-1].predict(self.x)
            lossSum = lambda gamma: np.sum(self._seLoss(self.y, self._F + gamma * pre_pred_loss))
            # find optimal step size in direction of steepest descent
            res = minimize(lossSum, 0.8, method='SLSQP')
            if "successfully." not in res.message:
                print(res.message)
            gamma_ = res.x[0]
            self._treeWeights.append(self.learning_rate * gamma_)
            self._F = self._F + gamma_ * self.learning_rate * pre_pred_loss
            # Compute Test Err if test data avalilbe
            if xTest is not None and yTest is not None:
                tstErr = self.testErr(xTest, yTest)
                print(" %4d     | %4e | %4e | %.3f" % (i, self.trainErr(),  tstErr, gamma_))
            else:
                print(" %4d     | %4e | %.3f " % (i, self.trainErr(), gamma_))


    def trainErr(self, loss='se'):
        """!
        @brief Training error at iteration m:
        \f[
        E_m = \sum_i L(F_{m_i} - y_i)
        \f]
        return (float) training error
        """
        return np.sum(self._seLoss(self.y, self._F)) / len(self.y)

    def testErr(self, xTest, yTest, loss='se'):
        return np.sum(self._seLoss(yTest, self.predict(xTest))) / len(yTest)

    def nablaLoss(self, y, yHat):
        """!
        @brief Jacobian of loss function. Computes:
        \f[
        -\nabla L(y, \hat y) = \frac{\sum_i{(y_i - f(x_i)})^2}{\partial f(x_i)}
        \f]
        Where \f[ f() \f] is the current model
        @param yHat  (1d_ndarray) predicted responses
        @param loss (str) Loss function name
        """
        if self.lossFn is 'se':
            return (y - yHat)
        else:
            raise NotImplementedError

    def _seLoss(self, y, yHat):
        """!
        @brief Squared error loss function.
        @param yHat  (1d_ndarray) predicted responses
        @return 1d_array  \f[ L = (y_i - \hat y_i) **2) \f]
        """
        # No need to divide by N here, outcome of minimization is the same
        return (y - yHat) ** 2.0


class ConstModel(object):
    def __init__(self, x, y, *args, **kwargs):
        self._yhat = np.mean(y)

    def predict(self, x, *args, **kwrags):
        return self._yhat * np.ones(len(x))
