#!/usr/bin/python3
from boosting.gbm import GBRTmodel
import matplotlib.pyplot as plt
import seaborn as sns
import unittest
import os
import numpy as np
pwd_ = os.path.dirname(os.path.abspath(__file__))
dataDir = pwd_ + "/data/"
np.random.seed(123)


def ground_truth(x):
    """!
    @brief Ground truth -- function to approximate
    Source:
    https://www.datarobot.com/blog/gradient-boosted-regression-trees
    by: Peter Prettenhofer
    """
    return x * np.sin(x) + np.sin(2 * x)

def gen_data(n_samples=200):
    """!
    @brief Generate training and testing data
    """
    np.random.seed(13)
    x = np.random.uniform(0, 10, size=n_samples)
    x.sort()
    y = ground_truth(x) + 0.75 * np.random.normal(size=n_samples)
    train_mask = np.random.randint(0, 2, size=n_samples).astype(np.bool)
    x_train, y_train = x[train_mask, np.newaxis], y[train_mask]
    x_test, y_test = x[~train_mask, np.newaxis], y[~train_mask]
    return x_train, x_test, y_train, y_test


class TestGradBoosting(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.xTrain, self.xTest, self.yTrain, self.yTest = \
           gen_data()

    def test1dBoostedReg(self):
        # In this case, use tree stumps for weak learners
        iters = 30
        gbt = GBRTmodel(maxTreeDepth=1, learning_rate=0.2, subsample=1.0)
        gbt.train(self.xTrain, self.yTrain, maxIterations=iters, xTest=self.xTest, yTest=self.yTest)

        # Eval 1d regression model
        xTest = np.linspace(0, 10, 400)
        yhat = gbt.predict(xTest)

        # plot
        plt.figure()
        plt.plot(self.xTrain, self.yTrain, marker='.', linestyle="None", label="Train Data")
        plt.plot(self.xTest, self.yTest, marker='.', linestyle="None", label="Test Data")
        plt.plot(xTest, yhat, label="Iter=" + str(iters))
        plt.legend(loc=0)
        plt.savefig('1d_boosted_regression_ex.png')
