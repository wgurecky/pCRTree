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
        # tree_weight * (weights * (T_i(x) == K))
        histCols = []
        lenX = testX.shape[0]
        for k in np.sort(np.unique(self.y)):
            counts = np.zeros(lenX)
            for i, tree in enumerate(self._trees):
                # where did this tree correcly predict?
                Ic = np.zeros(lenX)
                corrMask = (k == tree.predict(testX))
                # set diagonal to correctMask
                Ic[corrMask] = 1
                # corrPredict[np.diag_indices_from(corrPredict)] = Ic
                counts += self._treeWeights[i] * Ic
            histCols.append(counts)
        hist = np.array(histCols).T
        return hist

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
        for i in range(maxIterations):
            # subsample training data
            sub_idx = np.random.choice([True, False], len(y), p=[self.subsample, 1. - self.subsample])
            # Fit classification tree to training data with current weights
            self._trees.append(ClsTree(self.x[sub_idx], self.y[sub_idx],
                maxDepth=self.maxTreeDepth, weights=y_weights[sub_idx]))
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
            status.append([i, self.fracError(x, y), err])
            print(" %4d     | %4e | %.3f " % (status[i][0], status[i][1], status[i][2]))
        return np.array(status)

    def fracError(self, x, y):
        """!
        @brief Compute fraction of misclassified data points
        @return fraction of misclassified points
        """
        yhat = self.predict(x)
        corrMask = (yhat != y)
        return np.sum(corrMask) / len(y)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import make_gaussian_quantiles
    # run simple 2d classification tree example

    # Construct dataset
    X1, y1 = make_gaussian_quantiles(cov=2.,
                                     n_samples=100, n_features=2,
                                     n_classes=2, random_state=1)
    X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                     n_samples=400, n_features=2,
                                     n_classes=2, random_state=1)
    X = np.concatenate((X1, X2))
    y = np.concatenate((y1, - y2 + 1))

    # boosted Classification tree implementation
    bdt = GBCTmodel(maxTreeDepth=4, learning_rate=0.5, subsample=0.6)
    bdt.train(X, y, maxIterations=50)
    # SKlearn implementation
    skt = DecisionTreeClassifier(max_depth=5)
    skt.fit(X, y)

    plot_colors = "br"
    plot_step = 0.02
    class_names = "AB"

    plt.figure(figsize=(10, 5))

    # Plot the decision boundaries
    plt.subplot(111)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    # compute predicted descision boundaries
    Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z = skt.predict(np.c_[xx.ravel(), yy.ravel()])

    # Plot
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis("tight")

    # Plot the training points
    for i, n, c in zip(range(2), class_names, plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1],
                    c=c, cmap=plt.cm.Paired,
                    label="Class %s" % n, s=10)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Decision Boundary')

    plt.savefig("dblgauss_boosted_classify_ex.png")
    plt.close()
