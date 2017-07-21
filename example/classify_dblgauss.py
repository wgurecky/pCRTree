
#!/usr/bin/python3
##!
# \brief Performs two-class classification on a double gaussian
# distribution using boosted classification trees.
##
from boosting.gbc import GBCTmodel
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
pwd_ = os.path.dirname(os.path.abspath(__file__))
dataDir = pwd_ + "/data/"
np.random.seed(123)


def main():
    # Construct dataset
    X1, y1 = make_gaussian_quantiles(cov=2.,
                                     n_samples=100, n_features=2,
                                     n_classes=2, random_state=1)
    X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                     n_samples=400, n_features=2,
                                     n_classes=2, random_state=1)
    X = np.concatenate((X1, X2))
    y = np.concatenate((y1, - y2 + 1))

    # boosted Classification tree implementation
    bdt = GBCTmodel(maxTreeDepth=4, learning_rate=0.2, subsample=0.6)
    print("fitting pCRTree")
    bdt.train(X, y, maxIterations=100)

    # SKlearn implementation
    # print("fitting sklearn")
    # skt = DecisionTreeClassifier(max_depth=5)
    # skt.fit(X, y)

    plot_colors = ("aqua", "firebrick")
    plot_step = 0.02
    class_names = "AB"

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    print("predicting pCRTree")
    # compute predicted descision boundaries
    sample_locs = np.c_[xx.ravel(), yy.ravel()]
    Z = bdt.predict(sample_locs)

    # compare against sklearn
    # Z = skt.predict(np.c_[xx.ravel(), yy.ravel()])

    # Plot
    plt.figure(figsize=(6, 5))
    plt.subplot(111)

    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z,
            cmap=sns.diverging_palette(220, 20, sep=20, as_cmap=True))
    plt.axis("tight")

    # Plot the training points
    for i, n, c in zip(range(2), class_names, plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1],
                    c=c, cmap=plt.cm.Paired,
                    label="Class %s" % n, s=10)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Decision Boundary')
    plt.savefig("dblgauss_boosted_classify_ex.png")
    plt.close()

    plt.figure(figsize=(6.5, 5))
    plt.subplot(111)

    # Plot class label likelihood contourf
    class_probs = bdt.predictClassProbs(sample_locs)
    print("Max Prob %f" % np.max(class_probs))
    print("Min Prob %f" % np.min(class_probs))
    Z_0 = class_probs[:, 0].reshape(xx.shape)
    # Plot
    Z = Z_0
    cs = plt.contourf(xx, yy, Z, alpha=0.8, cmap="GnBu")
    cl = plt.contour(xx, yy, Z, 2, alpha=0.1, colors='k', hold="on", antialiased=True)
    plt.grid(b=True, which='major', linestyle='--', color='k')

    # Plot the training points
    for i, n, c in zip(range(2), class_names, plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1],
                    c=c, cmap=plt.cm.Paired,
                    label="Class %s" % n, s=10)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.colorbar(cs, ticks=[0, 0.5, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Class Blue Probability')
    plt.savefig("dblgauss_boosted_classify_probs.png")
    plt.close()

if __name__ == "__main__":
    main()
