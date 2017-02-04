#!/usr/bin/python3

import numpy as np
from node import BiNode

class RegTree(BiNode):
    """!
    @brief Regression tree
    """
    def __init__(self, x, y, yhat=None, level=0, maxDepth=3, minSplitPts=5):
        super().__init__(x, y, yhat, level, maxDepth, minSplitPts)

    def predict(self, testX):
        """!
        @brief Given some testing input, return CART tree
        predictions.
        Traverse the tree and provide leaf node predictions
        @param testX numpy nd_array of ints or floats
        """
        if len(np.shape(testX)) == 1:
            testX = np.array([testX]).T
        if testX.shape[1] != self.ndim:
            print("ERROR: dimension mismatch.")
            raise RuntimeError
        xHat, yHat = self.nodePredict(testX)
        return xHat, yHat

    def nodePredict(self, testX, xHat=np.array([[]]), yHat=np.array([])):
        if self._nodes != (None, None):
            leftX, _l, rightX, _r = self._maskData(self._spl, self._spd, testX)
            lxh, lyh = self._nodes[0].nodePredict(leftX, xHat, yHat)
            rxh, ryh = self._nodes[1].nodePredict(rightX, xHat, yHat)
            try:
                return np.vstack((xHat, lxh, rxh)), np.hstack((yHat, lyh, ryh))
            except:
                return np.vstack((lxh, rxh)), np.hstack((lyh, ryh))
        else:
            # is leaf node
            if xHat.shape[1] == 0:
                xHat = testX
                yHat = self._yhat * np.ones(len(testX))
            else:
                xHat = np.vstack((xHat, testX))
                yHat = np.hstack((yHat, self._yhat * np.ones(len(testX))))
            return xHat, yHat

    def _regionFit(self, region_x, region_y, lossFn="squared"):
        """!
        @brief Evaulate region loss fuction:
            - squared errors
            - L1 errors
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
    # run simple 1d regression tree example
    n = 100
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x)
    yNoise = np.random.uniform(0, 0.0001, n)
    y = y + yNoise
    regressionTree = RegTree(x, y)
    regressionTree.fitTree()

    # predict
    xTest = np.linspace(0, 2 * np.pi, n * 2)
    xhat, yhat = regressionTree.predict(xTest)

    # plot
    plt.figure()
    plt.plot(x, y, label="Train Data")
    plt.plot(xhat[:, 0], yhat, label="Reg Tree")
    plt.legend()
    plt.show()
