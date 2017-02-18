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
    bdt = GBCTmodel(maxTreeDepth=4, learning_rate=0.5, subsample=0.6)
    bdt.train(X, y, maxIterations=50)
    # SKlearn implementation
    skt = DecisionTreeClassifier(max_depth=5)
    skt.fit(X, y)

    plot_colors = ("b", "firebrick")
    plot_step = 0.02
    class_names = "AB"

    plt.figure(figsize=(10, 5))

    # Plot the decision boundaries
    plt.subplot(111)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    # compute predicted descision boundaries
    Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])

    # compare against sklearn
    # Z = skt.predict(np.c_[xx.ravel(), yy.ravel()])

    # Plot
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


if __name__ == "__main__":
    main()
