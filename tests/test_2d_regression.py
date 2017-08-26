#!/usr/bin/python3
from dtree.regress import RegTree
from scipy.interpolate import griddata
import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
from pylab import cm
pwd_ = os.path.dirname(os.path.abspath(__file__))
dataDir = pwd_ + "/data/"
np.random.seed(123)


class Test2dRegression(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """!
        @brief Generate 2d training data set.
        """
        x1 = np.linspace(0, 2 * np.pi, 100)
        x2 = np.linspace(0, 2 * np.pi, 100)
        X1, X2 = np.meshgrid(x1, x2)
        Z = flux_qbit_pot(X1, X2).T
        self.x = np.array([X1.flatten(), X2.flatten()]).T
        self.y = Z.flatten()

        # plot training data
        try:
            plt.figure(0)
            plt.pcolor(X1 / (np.pi * 2), X2 / (np.pi * 2), Z, cmap=cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max())
            plt.colorbar()
            plt.savefig('train_2d_regress.png')
            plt.close()
        except:
            pass

    def testRegression(self):
        """!
        @brief Test 2d non-boosted tree regression at high depth
        """
        # generate testing input
        x1 = np.linspace(0, 2 * np.pi, 50)
        x2 = np.linspace(0, 2 * np.pi, 50)
        X1, X2 = np.meshgrid(x1, x2)
        xTest = np.array([X1.flatten(), X2.flatten()]).T

        # fit 2d regression tree
        regTree2D = RegTree(self.x, self.y, maxDepth=12)
        regTree2D.fitTree()
        zHat = regTree2D.predict(xTest)

        # check min and max predictions
        self.assertTrue((np.abs((np.max(self.y) - np.max(zHat))) / np.mean(self.y) < 0.05))
        self.assertTrue((np.abs((np.min(self.y) - np.min(zHat))) / np.mean(self.y) < 0.05))

        # plot
        x1grid = np.linspace(xTest[:, 0].min(), xTest[:, 0].max(), 200)
        x2grid = np.linspace(xTest[:, 1].min(), xTest[:, 1].max(), 200)
        x1grid, x2grid = np.meshgrid(x1grid, x2grid)
        zgrid = griddata((xTest[:, 0], xTest[:, 1]), values=zHat, xi=(x1grid, x2grid), method='nearest')
        try:
            plt.figure(1)
            plt.pcolor(x1grid / (np.pi * 2), x2grid / (np.pi * 2), zgrid,
                       cmap=cm.RdBu, vmin=abs(zgrid).min(), vmax=abs(zgrid).max())
            plt.colorbar()
            plt.savefig('test_2d_tree_regress.png')
        except:
            pass


def flux_qbit_pot(phi_m, phi_p):
    alpha, phi_ext = 0.7, 2 * np.pi * 0.5
    return 2 + alpha - 2 * np.cos(phi_p) * np.cos(phi_m) - alpha * np.cos(phi_ext - 2 * phi_p)

