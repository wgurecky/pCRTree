#!/usr/bin/python3
##!
# @brief Binary CART tree node
# @author William Gurecky
# @date Feb 3 2017
##
import numpy as np
from numba import jit


class BiNode(object):
    """!
    @brief Binary node object.  Can be a leaf node,
    or can point to two other nodes.
    """
    def __init__(self, x, y, yhat=None, level=0, maxDepth=3, minSplitPts=4):
        """!
        @param x nd_array of integers or floats shape = (Npts, D)
        @param y 1d_array of integers or floats
        @param yhat (float) constant prediction value in node
        @param level int level of node in the tree
        @param maxDepth maximum number of levels in descission tree
        @param minSplitPts minimum number of points in node to be considered
            for further splitting.
        """
        # left and right node storage
        self._nodes = (None, None)
        self._split_gain = 0.
        self.level = level
        self.maxDepth = maxDepth
        self.minSplitPts = minSplitPts
        # Make x 2D array
        if len(x.shape) == 1:
            x = np.array([x]).T
        self.ndim = x.shape[1]
        if len(y.shape) != 1:
            raise RuntimeError("ERROR: Y data must be 1d")

        # node data
        self.x, self.y = x, y

        # predictor storage
        self._yhat = yhat
        self._spl, self._spd = None, None

    @property
    def spl(self):
        return self._spl

    @spl.setter
    def spl(self, spl):
        self._spl = spl

    @property
    def spd(self):
        return self._spd

    @spd.setter
    def spd(self, spd):
        self._spd = spd

    @property
    def yhat(self):
        return self._yhat

    @yhat.setter
    def yhat(self, ypredict):
        self._yhat = ypredict

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, Nleft=None, Nright=None):
        self._nodes = (Nleft, Nright)

    @property
    def split_gain(self):
        return self._split_gain

    @split_gain.setter
    def split_gain(self, gain):
        self._split_gain = gain

    def feature_importances_(self, **kwargs):
        """!
        @brief Recursively traverses tree and tallies split axis
        and split "benifit".
        @return  np_1darray of feature importances in this CART tree.
        """
        verbose = kwargs.pop("verbose", 0)
        tree_gain = kwargs.get("imp_arr", np.zeros(self.x.shape[1]))
        assert(len(tree_gain) == self.x.shape[1])
        if self._nodes != (None, None):
            # Note the gain we achived when splitting
            # and along what dimension we split
            node_gain = np.zeros(self.x.shape[1])
            node_gain[int(self._spd)] = self._split_gain
            # Add split gain to tree gain
            tree_gain += node_gain
            #
            tree_gain = self._nodes[0].feature_importances_(imp_arr=tree_gain)
            tree_gain = self._nodes[1].feature_importances_(imp_arr=tree_gain)
            if verbose:
                print("----------")
                print("node gains: " + \
                      str(node_gain) + " lvl: " + str(self.level))
            return tree_gain
        else:
            # leaf node has no splits
            return tree_gain

    def isLeaf(self):
        """!
        @brief Returns if this is a leaf or interior node.
        """
        if self._nodes == (None, None):
            return True
        else:
            return False

    def delData(self):
        """!
        @brief Deletes data in an interior node.
        """
        if self.isLeaf():
            pass
        else:
            del self.x
            del self.y

    def splitLocs(self):
        """!
        @brief Determine all possible split locations in all dimensions
        """
        testSplits = []
        for i, data in enumerate(self.x.T):
            suD = np.unique(np.sort(data))
            # compute all possible split locs
            splits = np.mean((suD[1:], suD[:-1]), axis=0)
            testSplits.append(splits)
        return testSplits

    def iterSplitData(self):
        """!
        @brief Generates split datasets
        """
        testSplits = self.splitLocs()
        for d in range(np.shape(self.x)[1]):
            for spl in testSplits[d]:
                leftExpl, leftData, rightExpl, rightData = \
                    maskDataJit(spl, d, self.x, self.y)
                yield ([leftExpl, leftData], [rightExpl, rightData], d, spl)

    def _maskData(self, spl, d, x, y=None):
        """!
        @brief Given split location and dimension along which to split,
        partition the data into left and right datasets.
        @param spl  Split location (int or float)
        @param d    Split dimension (int)
        @param x  Explanatory variables nd_array
        @param y  Response vars 1d_array
        """
        leftMask = (x[:, int(d)] < spl)
        # rightMask = (x[:, int(d)] >= spl)
        rightMask = np.invert(leftMask)
        leftExpl = x[leftMask]
        rightExpl = x[rightMask]
        if y is not None:
            leftData = y[leftMask]
            rightData = y[rightMask]
        else:
            leftData = None
            rightData = None
        return leftExpl, leftData, rightExpl, rightData

    @staticmethod
    def internalEvalSplit(split, node_err, gain_measure):
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
        return [eTot, vL, vR, split[2], split[3], gain]

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
        splitErrors = []
        nodeErr = regionFitJit(self.x, self.y)[0]
        # Internal Split Eval
        splitErrors = [self.internalEvalSplit(slt, nodeErr, gain_measure) \
                       for slt in self.iterSplitData()]
        #
        splitErrors = np.array(splitErrors)
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

    def fitTree(self):
        """!
        @brief Recursively grow the CART tree to fit data.
        """
        splitSuccess = self.splitNode()
        if not splitSuccess:
            return
        for childNode in self._nodes:
            childNode.fitTree()

    def _isGoodSplit(self):
        raise NotImplementedError

    def splitNode(self):
        raise NotImplementedError

    def _regionFit(self, explanData, resonseData):
        """!
        @brief Evaulate region loss fuction:
            - squared errors
        @return (err, bestFunctionValue)
        """
        raise NotImplementedError


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


@jit(nopython=True)
def maskDataJit(spl, d, x, y):
    """!
    @brief Given split location and dimension along which to split,
    partition the data into left and right datasets.
    @param spl  Split location (int or float)
    @param d    Split dimension (int)
    @param x  Explanatory variables nd_array
    @param y  Response vars 1d_array
    """
    leftMask = (x[:, int(d)] < spl)
    rightMask = np.invert(leftMask)
    leftExpl = x[leftMask]
    rightExpl = x[rightMask]
    leftData = y[leftMask]
    rightData = y[rightMask]
    return leftExpl, leftData, rightExpl, rightData
