#!/usr/bin/python3
##
# \brief Implements regression trees using greedy,
# best first splits.
##
import numpy as np
from dtree.node import BiNode


class RegTree(BiNode):
    """!
    @brief Regression tree
    """
    def __init__(self, x, y, yhat=None, level=0, maxDepth=3, minSplitPts=5):
        """!
        @param x nd_array of integers or floats shape = (Npts, D)
        @param y 1d_array of integers or floats
        @param yhat (float) constant prediction value in node
        @param level int level of node in the tree
        @param maxDepth maximum number of levels in descission tree
        @param minSplitPts minimum number of points in node to be considered
            for further splitting.
        """
        super(RegTree, self).__init__(x, y, yhat, level, maxDepth, minSplitPts)

    def predict(self, testX):
        """!
        @brief Given some testing input return regression tree predictions.
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
            leftX, lIdX, rightX, rIdX = self._maskData(self._spl, self._spd, testX, testXIdx)
            lxh, lyh, lIdx = self._nodes[0].bNodePredict(leftX, lIdX)
            rxh, ryh, rIdx = self._nodes[1].bNodePredict(rightX, rIdX)
            return np.vstack((lxh, rxh)), np.hstack((lyh, ryh)), np.hstack((lIdx, rIdx))
        else:
            # is leaf node
            xHat = testX
            yHat = self._yhat * np.ones(len(testX))
            return xHat, yHat, testXIdx

    def _regionFit(self, region_x, region_y, lossFn="squared"):
        """!
        @brief Evaulate region loss fuction:
            - squared errors
        @return (loss, regionYhat)
        """
        yhat = np.mean(region_y)
        # residual sum squared error
        rsse = np.sum((region_y - yhat) ** 2)
        return rsse, yhat

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
            splitData = self._maskData(spl, d, self.x, self.y)

            # store split location and split dimension on current node
            self._spl = spl
            self._spd = d

            # create left and right child nodes
            leftNode = RegTree(splitData[0], splitData[1], lYhat, self.level + 1, self.maxDepth)
            rightNode = RegTree(splitData[2], splitData[3], rYhat, self.level + 1, self.maxDepth)
            self._nodes = (leftNode, rightNode)
            if cleanUp: self.delData()
            return 1


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    # run simple 1d regression tree example
    n = 100
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x)
    yNoise = np.random.uniform(0, 0.1, n)
    y = y + yNoise
    regressionTree = RegTree(x, y, maxDepth=4)
    regressionTree.fitTree()
    regressionTree3 = RegTree(x, y, maxDepth=3)
    regressionTree3.fitTree()

    # predict
    xTest = np.linspace(0, 2 * np.pi, n * 2)
    yhat = regressionTree.predict(xTest)
    yhat3 = regressionTree3.predict(xTest)

    # plot
    plt.figure()
    plt.plot(x, y, label="Train Data")
    plt.plot(xTest, yhat, label="Tree Depth=4")
    plt.plot(xTest, yhat3, label="Tree Depth=3")
    plt.legend()
    plt.savefig('1d_regression_ex.png')
