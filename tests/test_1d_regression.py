#!/usr/bin/python3
from dtree.regress import RegTree
import unittest
import os
import numpy as np
pwd_ = os.path.dirname(os.path.abspath(__file__))
dataDir = pwd_ + "/data/"
np.random.seed(123)


class Test1dRegression(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.n = 100
        self.x = np.linspace(0, 2 * np.pi, self.n)
        y = np.sin(self.x)
        yNoise = np.random.uniform(0, 0.1, self.n)
        self.y = y + yNoise

    def testRegression(self):
        regressionTree = RegTree(self.x, self.y)
        regressionTree.fitTree()

        # predict
        xTest = np.linspace(0, 2 * np.pi, self.n * 2)
        xhat, yhat = regressionTree.predict(xTest)

        # check result
        self.assertEqual(len(xhat), len(yhat))
        self.assertEqual(self.n * 2, len(yhat))
        self.assertAlmostEqual(np.mean(yhat), np.mean(self.y), delta=0.1)

    def testTreeDepth(self):
        """!
        @brief Tests adjusting the maximum tree depth
        """
        # test inputs
        xTest = np.linspace(0, 2 * np.pi, self.n)

        rd2Tree = RegTree(self.x, self.y, maxDepth=2)
        rd2Tree.fitTree()
        xhat2, yhat2 = rd2Tree.predict(xTest)
        rd2TreeErr = np.linalg.norm(yhat2 - self.y)
        #
        rd3Tree = RegTree(self.x, self.y, maxDepth=3)
        rd3Tree.fitTree()
        xhat3, yhat3 = rd3Tree.predict(xTest)
        rd3TreeErr = np.linalg.norm(yhat3 - self.y)
        #
        rd4Tree = RegTree(self.x, self.y, maxDepth=4)
        rd4Tree.fitTree()
        xhat4, yhat4 = rd4Tree.predict(xTest)
        rd4TreeErr = np.linalg.norm(yhat4 - self.y)

        # A larger tree depth should reduce squared errors of predictor
        self.assertTrue(rd2TreeErr > rd3TreeErr)
        self.assertTrue(rd3TreeErr > rd4TreeErr)

    def testOverfit(self):
        """!
        @brief Specify a tree with large depth relative to number
        of training data points.
        """
        overFitTree = RegTree(self.x, self.y, maxDepth = 100)
