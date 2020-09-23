#!/usr/bin/python3
##
# \brief Implements regression trees using greedy,
# best first splits.
##
from __future__ import division
import numpy as np
from numba import jit
from dtree.node import BiNode, maskDataJit


class RegTree(BiNode):
    """!
    @brief Regression tree
    """
    def __init__(self, x, y, yhat=None, level=0, maxDepth=3, minSplitPts=8, minDataLeaf=4, **kwargs):
        """!
        @param x nd_array of integers or floats shape = (Npts, D)
        @param y 1d_array of integers or floats
        @param yhat (float) constant prediction value in node
        @param level int level of node in the tree
        @param maxDepth maximum number of levels in descission tree
        @param minSplitPts minimum number of points in node to be considered
            for further splitting.
        @param minDataLeaf minimum number of points required to form a new node
        """
        super(RegTree, self).__init__(x, y, yhat, level, maxDepth, minSplitPts, minDataLeaf)

    def _regionFit(self, region_x, region_y, lossFn="squared"):
        """!
        @brief Evaulate region squared error loss fuction
        @return (loss, regionYhat)
        """
        yhat = np.mean(region_y)
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

    def splitNode(self, cleanUp=False):
        """!
        @brief Partition the data in the current node.
        Create new nodes with partitioned data sets.
        """
        if self._isGoodSplit() is False:
            return 0
        else:
            bs = self.evalSplits()
            if bs is None:
                return 0
            lYhat = bs[1]
            rYhat = bs[2]
            d, spl = bs[3], bs[4]
            splitData = maskDataJit(spl, d, self.x, self.y)

            # store split location and split dimension on current node
            self._spl = spl
            self._spd = d
            self._split_gain = bs[5]

            # create left and right child nodes
            leftNode = RegTree(splitData[0], splitData[1], lYhat, self.level + 1, self.maxDepth, minSplitPts=self.minSplitPts)
            rightNode = RegTree(splitData[2], splitData[3], rYhat, self.level + 1, self.maxDepth, minSplitPts=self.minSplitPts)
            self._nodes = (leftNode, rightNode)
            if cleanUp: self.delData()
            return 1

    def evalSplits(self, split_crit="best", gain_measure="se"):
        """!
        @brief evaluate loss function in each split region
        @param split_crit str in ("best", "var"):
            Splits on sum sqr err or varience reduction criteria respectively.
        @param gain_measure  Measure by which split gain is computed
            "se": squared error
            "var": varience improvement
        @return list [totError, valLeft, valRight, split_dimension, split_loc]
        """
        nodeErr = regionFitJit(self.x, self.y)[0]
        # Internal Split Eval
        splitErrors = np.array([self.internalEvalSplit(slt, nodeErr, gain_measure) \
                                for slt in self.iterSplitData()])
        if splitErrors.size == 0:
            return None
        if split_crit is "best":
            # split on sum squared err
            best_gain = np.min(splitErrors[:, 0])
            tie_mask = (splitErrors[:, 0] == best_gain)
            n_ties = np.count_nonzero(tie_mask)
            if n_ties >= 2:
                candidateIdxs = np.nonzero(tie_mask)[0]
                bestSplitIdx = np.random.choice(candidateIdxs)
            else:
                bestSplitIdx = np.argmin(splitErrors[:, 0])
        else:
            # split on gain
            best_gain = np.max(splitErrors[:, 5])
            tie_mask = (splitErrors[:, 5] == best_gain)
            n_ties = np.count_nonzero(tie_mask)
            if n_ties >= 2:
                candidateIdxs = np.nonzero(tie_mask)[0]
                bestSplitIdx = np.random.choice(candidateIdxs)
            else:
                bestSplitIdx = np.argmax(splitErrors[:, 5])
        # select the best possible split
        return splitErrors[bestSplitIdx]

    def internalEvalSplit(self, split, node_err, gain_measure):
        eL, vL = regionFitJit(split[0][0], split[0][1])
        eR, vR = regionFitJit(split[1][0], split[1][1])
        eTot = eL + eR
        if gain_measure == "se":
            gain = eTot
        else:
            # reduction in varience from split
            n = len(self.y)
            n_l = len(split[0][1])
            n_r = len(split[1][1])
            node_var = (1. / n) * 0.5 * node_err
            split_var = (1. / n_l) * 0.5 * eL + \
                        (1. / n_r) * 0.5 * eR
            gain = node_var - split_var
        if len(split[0][1]) < self.minDataLeaf or \
           len(split[1][1]) < self.minDataLeaf:
            return [eTot+1e20, vL, vR, split[2], split[3], gain-1e20]
        else:
            return [eTot, vL, vR, split[2], split[3], gain]


class RegTreeLin(RegTree):
    def __init__(self, x, y, yhat=None, level=0, maxDepth=3, minSplitPts=8, minDataLeaf=4,
                 yhat_slope=None, yhat_intercept=None, spd=None, **kwargs):
        super(RegTreeLin, self).__init__(x, y, yhat, level, maxDepth, minSplitPts, minDataLeaf)
        self._yhat_slope = yhat_slope
        self._yhat_intercept = yhat_intercept
        self._spd = spd

    def fitTree(self):
        super(RegTreeLin, self).fitTree()

    def splitNode(self, cleanUp=False):
        """!
        @brief Partition the data in the current node.
        Create new nodes with partitioned data sets.
        """
        if self._isGoodSplit() is False:
            return 0
        else:
            bs = self.evalSplits()
            if bs is None:
                return 0
            lYhat = bs[1]
            rYhat = bs[2]
            d, spl = bs[3], bs[4]
            splitData = maskDataJit(spl, d, self.x, self.y)

            # store split location and split dimension on current node
            self._spl = spl
            self._spd = d
            self._split_gain = bs[5]

            _, l_yhat_slope, l_yhat_intercept = regionFitLin(splitData[0], splitData[1], int(self._spd))
            _, r_yhat_slope, r_yhat_intercept = regionFitLin(splitData[2], splitData[3], int(self._spd))

            # create left and right child nodes
            leftNode = RegTreeLin(splitData[0], splitData[1], lYhat, self.level + 1, self.maxDepth, \
                    yhat_slope=l_yhat_slope, yhat_intercept=l_yhat_intercept, spd=self._spd, minSplitPts=self.minSplitPts)
            rightNode = RegTreeLin(splitData[2], splitData[3], rYhat, self.level + 1, self.maxDepth, \
                    yhat_slope=r_yhat_slope, yhat_intercept=r_yhat_intercept, spd=self._spd, minSplitPts=self.minSplitPts)
            self._nodes = (leftNode, rightNode)
            if cleanUp: self.delData()
            return 1


# ============================NUMBA FUNCTIONS================================ #
@jit(nopython=True)
def regionFitJit(region_x, region_y):
    """!
    @brief Evaulate region loss fuction
    (Squared errors).  Return residual sum squared
    errors and the region prediction (mean).
    @param region_x np_ndarray predictors in this region
    @param region_y np_1darray responses in this region
    @return (loss, regionYhat)
    """
    yhat = np.mean(region_y)
    rsse = np.sum((region_y - yhat) ** 2)
    return rsse, yhat


def linregress_jit(x, y):
    # construct vandermonde matrix
    A = np.vstack([x, np.ones(len(x))]).T
    # solve overconstrained problem in least squares sense
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c


def regionFitLin(region_x, region_y, dsplit):
    # fit slope only for region that was split on before
    yhat_slope, yhat_intercept = linregress_jit(region_x[:,int(dsplit)], region_y)
    rsse = np.sum((region_y - (yhat_slope * region_x[:,int(dsplit)] + yhat_intercept)) ** 2)
    return rsse, yhat_slope, yhat_intercept
