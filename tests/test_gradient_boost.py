#!/usr/bin/python3
from boosting.gbm import GBRTmodel
from scipy.interpolate import griddata
from pylab import cm
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

        #2d data
        x1 = np.linspace(0, 2 * np.pi, 100)
        x2 = np.linspace(0, 2 * np.pi, 100)
        X1, X2 = np.meshgrid(x1, x2)
        Z = flux_qbit_pot(X1, X2).T
        self.x = np.array([X1.flatten(), X2.flatten()]).T
        self.y = Z.flatten()

    def test1dBoostedReg(self):
        # In this case, use tree stumps for weak learners
        iters = 80
        gbt = GBRTmodel(maxTreeDepth=1, learning_rate=0.1, subsample=0.7)
        gbt.train(self.xTrain, self.yTrain, maxIterations=iters, xTest=self.xTest, yTest=self.yTest)

        # Eval 1d regression model
        xTest = np.linspace(0, 10, 400)
        yhat = gbt.predict(xTest)

        # plot
        plt.figure(1)
        plt.plot(self.xTrain, self.yTrain, marker='.', linestyle="None", label="Train Data")
        plt.plot(self.xTest, self.yTest, marker='.', linestyle="None", label="Test Data")
        plt.plot(xTest, yhat, label="Iter=" + str(iters))
        plt.legend(loc=0)
        plt.savefig('1d_boosted_regression_ex.png')
        plt.close()

    def test2dBoostedReg(self):
        iters = 40
        gbt = GBRTmodel(maxTreeDepth=2, learning_rate=0.2, subsample=0.5)

        # generate testing input
        x1 = np.linspace(0, 2 * np.pi, 50)
        x2 = np.linspace(0, 2 * np.pi, 50)
        X1, X2 = np.meshgrid(x1, x2)
        xTest = np.array([X1.flatten(), X2.flatten()]).T

        # fit 2d boosted regression tree
        gbt.train(self.x, self.y, maxIterations=iters)
        zHat = gbt.predict(xTest)

        # plot
        x1grid = np.linspace(xTest[:, 0].min(), xTest[:, 0].max(), 200)
        x2grid = np.linspace(xTest[:, 1].min(), xTest[:, 1].max(), 200)
        x1grid, x2grid = np.meshgrid(x1grid, x2grid)
        zgrid = griddata((xTest[:, 0], xTest[:, 1]), values=zHat, xi=(x1grid, x2grid), method='nearest')
        plt.figure(2)
        plt.pcolor(x1grid / (np.pi * 2), x2grid / (np.pi * 2), zgrid, cmap=cm.RdBu, vmin=abs(zgrid).min(), vmax=abs(zgrid).max())
        plt.colorbar()
        plt.savefig('2d_boosted_regression_ex.png')
        plt.close()


def flux_qbit_pot(phi_m, phi_p):
    alpha, phi_ext = 0.7, 2 * np.pi * 0.5
    return 2 + alpha - 2 * np.cos(phi_p) * np.cos(phi_m) - alpha * np.cos(phi_ext - 2 * phi_p)
