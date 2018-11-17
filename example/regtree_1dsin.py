#!/usr/bin/python3
##!
# \brief Trains a single regression tree on a
# sin() curve (single dimension).
##
from dtree.regress import RegTree
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette(sns.diverging_palette(250, 15, s=70, l=45, n=3, center="light"))
sns.set_color_codes()

def main():
    # run simple 1d regression tree example
    n = 100
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x)
    yNoise = np.random.uniform(0, 0.1, n)
    y = y + yNoise
    regressionTree = RegTree(x, y, maxDepth=4)
    regressionTree.fitTree()
    regressionTree2 = RegTree(x, y, maxDepth=2)
    regressionTree2.fitTree()
    regressionTree1 = RegTree(x, y, maxDepth=1)
    regressionTree1.fitTree()

    # predict
    xTest = np.linspace(0, 2 * np.pi, n * 2)
    yhat = regressionTree.predict(xTest)
    yhat2 = regressionTree2.predict(xTest)
    yhat1 = regressionTree1.predict(xTest)

    # plot
    plt.figure()
    plt.scatter(x, y, label="Train Data", s=4, c='k')
    plt.plot(xTest, yhat, label="Tree Depth=4", c='k')
    plt.plot(xTest, yhat2, label="Tree Depth=2", c='r', ls='--')
    plt.plot(xTest, yhat1, label="Tree Depth=1", c='b', ls='-.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(alpha=0.5, ls='--')
    plt.legend()
    plt.savefig('1d_regression_ex.png')


if __name__ == "__main__":
    main()
