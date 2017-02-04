from node import BiNode

class RegTree(BiNode):
    """!
    @brief Regression tree
    """
    def __init__(self, x, y, yhat=None, level=0, minSplitPts=5):
        super().__init__(x, y, yhat, level, minSplitPts)

    def predict(self, testX):
        """!
        @brief Given some testing input, return CART tree
        predictions.
        Traverse the tree and provide leaf node predictions
        @param testX numpy nd_array of ints or floats
        """
        if len(np.shape(testX)) == 0:
            testX = np.array([testX]).T
        if testX.shape[1] != self.ndim:
            print("ERROR: dimension mismatch.")
            raise RuntimeError
        self.testX = []
        self.testY = []
        self.nodePredict(testX)
        return self.testX, self.testY

    def nodePredict(self, testX):
        if self._nodes != (None, None):
            leftX, _l, rightX, _r = self._maskData(self._spl, self._spd, testX)
            self._nodes[0].nodePredict(leftX)
            self._nodes[1].nodePredict(rightX)
        else:
            self.testX.append(testX)
            self.testY.append(self._yhat * np.ones(len(testX)))
            return

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
        if len(self.y) >= self.minSplitPts and self.level <= self.maxDepth:
            return True
        else:
            return False

    def splitNode(self):
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
            return 1
