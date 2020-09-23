#!/usr/bin/python3
##!
# \brief Gradient boosted regression trees
# \date Feb 3 2017
# \author William Gurecky
##
import numpy as np
from itertools import islice
from scipy.optimize import minimize
from dtree.regress import RegTree, RegTreeLin
from boosting.loss import FLoss
from sklearn.preprocessing import StandardScaler


class GBRTmodel(object):
    """!
    @brief Gradient boosted regression tree model.

    Implemented regularization techniques:
    - shrinkage via tunable learning rate
    - stochastic gradient boosting

    Implemented loss functions:
    - huber
    - squared-error
    """
    def __init__(self, max_depth=3, learning_rate=1.0, subsample=1.0, loss='se', alpha=0.5, minSplitPts=4, minDataLeaf=2, **kwargs):
        """!
        @param max_depth  Maximum depth of each weak learner in the model.
            Equal to number of possible interactions captured by each tree in the GBRT.
        @param learning_rate  Scale the influence of each tree in model.
        @param subsample  Fraction of avalible data used for training in any given
            boosted iteration.
        @param minSplitPts minimum number of points in node to be considered
            for further splitting in the tree based weak learners
        @param minDataLeaf minimum number of points required to form a new node in
            a tree based weak learner
        @param loss  (str) Target function to minimize at each iteration of boosting
            string in ("se", "huber", "quantile")
        @param alpha  (float)  target quantile value.
            Only used for "quantile" loss, otherwise this parameter is ignored.
        """
        self.max_depth = max_depth
        self.minSplitPts = minSplitPts
        self.minDataLeaf = minDataLeaf
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.trans = StandardScaler()

        # internal storage (write out to external file on request)
        self._trees = [None]
        self._treeWeights = [1.0]
        self._F = None

        # training data
        self.x = np.array([[]])
        self.y = np.array([])

        # normalization
        self._scale = kwargs.get("scale", False)

        # Loss class instance
        self._alpha = alpha
        self._L = FLoss(loss, tau=self._alpha)

        # Weak learner method
        self.tree_method = kwargs.get("tree_method", "cart")

    def predict(self, testX):
        """!
        @brief Evaluate gradient boosted regression tree model.
        @param testX (nd_array) evaluate model at these points
        @return np_1darry model predictions at testX
        """
        # get final prediciton in boosted predictions iterable
        sum_trees_prediction = list(islice(self.staged_predict(testX),
                                           self.n_estimators-1,
                                           self.n_estimators))
        return sum_trees_prediction[0]

    def staged_predict(self, testX, **kwargs):
        """!
        @brief Boosted tree model generator.  Generatees truncated
        boosted models including up to ntree_limit trees.
        @param testX (nd_array) evaluate model at these points
        @return iterable
        """
        n_estimators = kwargs.get("ntree_limit", self.n_estimators)
        if len(np.shape(testX)) == 1:
            testX = np.array([testX]).T
        fHat = np.zeros(testX.shape[0])
        for weight, tree in zip(self._treeWeights[:n_estimators], self._trees[:n_estimators]):
            fHat += weight * tree.predict(testX)
            if self._scale:
                out = self.trans.inverse_transform(fHat.reshape(-1, 1))[: ,0]
                print(out)
                yield out
            else:
                yield fHat

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, sc):
        if sc:
            self._scale = sc
        else:
            self._scale = False

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        assert(alpha <= 1.)
        assert(alpha >= 0.)
        self._L.tau = alpha

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

    def genRegTree(self, *args, **kwargs):
        if self.tree_method == "cart":
            return RegTree(*args, **kwargs)
        elif self.tree_method == "lin":
            return RegTreeLin(*args, **kwargs)
        else:
            raise NotImplementedError("Weak learner type not implemented")

    def train(self, x, y, n_estimators=5, warmStart=0, **kwargs):
        """!
        @brief Train the regression tree model by the gradient boosting
        method.
        Initilize model with constant value:
        \f[
        F_{m=0} = mean(y)
        \f]
        Where y is the input traing data response vector.

        At each iteration fit weak learner (\f$ h_m \f$) to pseudo-residuals
        (Squared error loss function shown for example).
        \f[
        r(x)_m = -\nabla L(y, \hat y) = \frac{\partial \sum_i{(y_i - f(x_i)})^2}{\partial f(x_i)}
        \f]

        Next, seek to minimize w.r.t. \f$\gamma_m\f$:
        \f[
        \frac{\partial \sum_i( L(y_i, F_m + \gamma_m h(x_i)_m)) }{\partial \gamma_m} = 0
        \f]

        Update the model:
        \f[
        F_{m} = F_{m-1} + \gamma_m h_m
        \f]
        @param x (nd_array) Explanatory variable set.  Shape = (N_support_pts, Ndims)
        @param y (1d_array) Response variable vector. Shape = (N_support_pts,)
        @param n_estimators  Max number of boosted iterations.
        """
        xTest = kwargs.pop("xTest", None)
        yTest = kwargs.pop("yTest", None)
        status = []
        if xTest is not None and yTest is not None:
            print("Iteration | Training Err | Testing Err  | Tree weight ")
            print("======================================================")
        else:
            print("Iteration | Training Err | Tree weight ")
            print("========================================")
        # Reset model
        self.x = x
        if self._scale:
            self.y = self.trans.fit_transform(y.reshape(-1, 1))[:, 0]
        else:
            self.y = y
        self._trees = [ConstModel(x, y)]
        self._treeWeights = [1.0]
        self._F = self._trees[0].predict(self.x)
        for i in range(n_estimators):
            # def sub sample training set
            sub_idx = np.random.choice([True, False], len(y), p=[self.subsample, 1. - self.subsample])
            # fit learner to pseudo-residuals
            self._trees.append(self.genRegTree(self.x[sub_idx],
                                               self._L.gradLoss(self.y[sub_idx], self._F[sub_idx]),
                                               maxDepth=self.max_depth,
                                               minSplitPts=self.minSplitPts,
                                               minDataLeaf=self.minDataLeaf
                                               )
                              )
            self._trees[-1].fitTree()
            # define minimization problem
            pre_pred_loss = self.trees[-1].predict(self.x)
            lossSum = lambda gamma: np.sum(self._L.loss(self.y, self._F + gamma * pre_pred_loss))
            # find optimal step size in direction of steepest descent
            res = minimize(lossSum, 0.8, method='SLSQP')
            if "successfully" not in res.message:
                print(res.message)
            gamma_ = res.x[0]
            self._treeWeights.append(self.learning_rate * gamma_)
            self._F = self._F + gamma_ * self.learning_rate * pre_pred_loss
            # Compute Test Err if test data avalilbe
            trainErr = self.trainErr()
            if xTest is not None and yTest is not None:
                tstErr = self.testErr(xTest, yTest)
                print(" %4d     | %4e | %4e | %.3f" % (i, trainErr,  tstErr, gamma_))
                status.append([i, trainErr, tstErr, gamma_])
            else:
                print(" %4d     | %4e | %.3f " % (i, trainErr, gamma_))
                status.append([i, trainErr, gamma_])
        return np.array(status)


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

    @property
    def n_estimators(self):
        return len(self._trees)

    @property
    def feature_importances_(self):
        total_sum = np.zeros(np.shape(self.x)[1])
        for weight, tree in zip(self._treeWeights[1:], self._trees[1:]):
            tree_importance = tree.feature_importances_()
            total_sum += tree_importance * 1.0
        print("*** TOTAL SUM IMPORTANCES ***")
        print(total_sum)
        importances = total_sum / np.sum(self._treeWeights)
        return importances / np.sum(importances)


class ConstModel(object):
    def __init__(self, x, y, *args, **kwargs):
        self._yhat = np.mean(y)

    def predict(self, x, *args, **kwrags):
        return self._yhat * np.ones(len(x))
