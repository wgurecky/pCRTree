#!/usr/bin/python3
##!
# \brief Trains a single regression tree on a
# sin() curve (single dimension).
##
from dtree.regress import RegTree
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
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


if __name__ == "__main__":
    main()
