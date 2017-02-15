#!/usr/bin/python3
##
# \brief Implements classification trees for partitioning an
# input space using a greedy, best-first fit.
# For a classification tree, the prediced output (yhat) in a given region
# is a class label (not a real number, as in a reg tree)
##
import numpy as np
import math
from dtree.node import BiNode


class ClsTree(BiNode):
    """!
    @brief Classification tree
    """
    def __init__(self, x, y, yhat=None, level=0, maxDepth=3, minSplitPts=5, weights=None):
        """!
        @param x nd_array of integers or floats. shape = (Npts, D)
        @param y 1d_array of integers. shape = (Npts,)
        @param yhat (int) constant prediction value in node
        @param level int level of node in the tree
        @param maxDepth maximum number of levels in descission tree
        @param minSplitPts minimum number of points in node to be considered
            for further splitting.
        """
        if ('int' not in str(y.dtype)):
            print("ERROR: Recast response variables to type int before classification.")
            raise TypeError
        super(ClsTree, self).__init__(x, y, yhat, level, maxDepth, minSplitPts)
        if weights is not None:
            self.weights = np.ones(len(y))
        else:
            self.weights = np.ones(len(y))
        self._nodeEr = self._regionFit(x, y, self._weights)[0]

    @property
    def weights(self):
        """!
        @brief Weights array accessor
        """
        return self._weights

    @weights.setter
    def weights(self, weights):
        """!
        @brief Set weights and ensure weights sum to 1
        @param weights 1d_array of weights corrosponding to y
        """
        if len(weights) != len(self.y):
            raise RuntimeError
        self._weights = weights
        self._weights /= np.sum(weights)

    def predict(self, testX):
        """!
        @brief Given some testing input return classification tree predictions.
        Traverses the tree recursively and provides leaf node predictions.
        @param testX nd_array of ints or floats.  Test explanatory input array
        @return 1d_array y_hat (len=len(testX)) prediction array
        """
        if len(np.shape(testX)) == 1:
            testX = np.array([testX]).T
        if testX.shape[1] != self.ndim:
            print("ERROR: dimension mismatch.")
            raise RuntimeError
        # xHat, yHat = self.nodePredict(testX)
        oIdx = np.arange(len(testX))
        xHat, yHat, xIdx = self.bNodePredict(testX, np.arange(len(testX)))
        if not np.array_equal(testX[xIdx], xHat):
            print("WARNING: Shifted output order!")
        shift = np.lexsort((oIdx, xIdx))
        return yHat[shift]

    def bNodePredict(self, testX, testXIdx):
        """!
        @brief Recursively evaluate internal node splits and leaf predictions.
        @return (xOut, yOut, xIdx_Out)
            indicies of original X vector and corrosponding resopnse Y
        """
        if self._nodes != (None, None):
            leftX, lIdX, rightX, rIdX, _1, _2 = self._maskData(self._spl, self._spd, testX, testXIdx)
            lxh, lyh, lIdx = self._nodes[0].bNodePredict(leftX, lIdX)
            rxh, ryh, rIdx = self._nodes[1].bNodePredict(rightX, rIdX)
            return np.vstack((lxh, rxh)), np.hstack((lyh, ryh)), np.hstack((lIdx, rIdx))
        else:
            # is leaf node
            xHat = testX
            yHat = self._yhat * np.ones(len(testX))
            return xHat, yHat, testXIdx

    def _regionFit(self, region_x, region_y, region_weights):
        """!
        @brief Evaulate region loss fuction:
            - Gini impurity
        optimal value is equal to the most likely value in the region
        of interest. In this case this is equal to the mode.
        @return (loss, regionYhat)
        """
        Er = 0.
        yhat = np.bincount(region_y).argmax()
        uq = np.unique(region_y)
        for u in uq:
            wgts = region_weights[(region_y == u)]
            p = np.sum(wgts) / len(region_y)
            # old unweighted frac
            # p = len(region_y[(region_y == u)]) / len(region_y)
            Er += -p * np.log2(p)
        return Er, yhat

    def evalSplits(self, split_crit="best"):
        """!
        @brief evaluate loss function in each split region
        @return list [totError, valLeft, valRight, split_dimension, split_loc]
        """
        splitErrors = []
        for split in self.iterSplitData():
            eL, vL = self._regionFit(split[0][0], split[0][1], split[0][2])
            eR, vR = self._regionFit(split[1][0], split[1][1], split[1][2])
            p = len(split[0][0]) / len(self.y)  # frac of points in left region
            gain = self._nodeEr - p * eL - (1 - p) * eR
            eTot = eL + eR
            splitErrors.append([eTot, vL, vR, split[2], split[3], gain])
        splitErrors = np.array(splitErrors)
        bestSplitIdx = np.argmax(splitErrors[:, 5])
        # select the best possible split
        return splitErrors[bestSplitIdx]

    def iterSplitData(self):
        """!
        @brief Generates split datasets
        """
        testSplits = self.splitLocs()
        for d in range(np.shape(self.x)[1]):
            for spl in testSplits[d]:
                leftExpl, leftData, rightExpl, rightData, leftWeights, rightWeights = \
                    self._maskData(spl, d, self.x, self.y, self._weights)
                yield ([leftExpl, leftData, leftWeights], [rightExpl, rightData, rightWeights], d, spl)

    def _maskData(self, spl, d, x, y=None, w=None):
        """!
        @brief Given split location and dimension along which to split,
        partition the data into left and right datasets.
        @param spl  Split location (int or float)
        @param d    Split dimension (int)
        @param x  Explanatory variables nd_array
        @param y  Response vars 1d_array
        """
        leftMask = (x[:, int(d)] < spl)
        rightMask = (x[:, int(d)] >= spl)
        leftExpl = x[leftMask]
        rightExpl = x[rightMask]
        if y is not None:
            leftData = y[leftMask]
            rightData = y[rightMask]
        else:
            leftData = None
            rightData = None
        if w is not None:
            leftWeights = w[leftMask]
            rightWeights = w[rightMask]
        else:
            leftWeights = None
            rightWeights = None
        return leftExpl, leftData, rightExpl, rightData, leftWeights, rightWeights

    def _isGoodSplit(self):
        """!
        @brief Evaluates if split is favorable (or possible under provided
        stopping criteria)
        """
        if len(self.y) >= self.minSplitPts and self.level < self.maxDepth:
            return True
        else:
            return False

    def splitNode(self, cleanUp=True):
        """!
        @brief Partition the data in the current node.
        Create new nodes with partitioned data sets.
        """
        if self._isGoodSplit() is False:
            return 0
        else:
            bs = self.evalSplits()
            lYhat = bs[1]
            rYhat = bs[2]
            d, spl = bs[3], bs[4]
            splitData = self._maskData(spl, d, self.x, self.y, self._weights)

            # store split location and split dimension on current node
            self._spl = spl
            self._spd = d

            # create left and right child nodes
            leftNode = ClsTree(splitData[0], splitData[1], lYhat, self.level + 1, self.maxDepth, weights=splitData[4])
            rightNode = ClsTree(splitData[2], splitData[3], rYhat, self.level + 1, self.maxDepth, weights=splitData[5])
            self._nodes = (leftNode, rightNode)
            print("Split at: %f in dimension: %d, yhat_left: %d, yhat_right: %d, level: %d" % (spl, d, lYhat, rYhat, self.level))
            return 1


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

    # pCRTree implementation
    bdt = ClsTree(X, y, maxDepth=10, minSplitPts=5)
    bdt.fitTree()
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

    plt.savefig("2d_classify_ex.png")
    plt.close()
