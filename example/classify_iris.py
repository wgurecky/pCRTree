#!/usr/bin/python3
##!
# \brief Performs multi-class classification on the iris dataset
# using the GBCTmodel.
##
from boosting.gbc import GBCTmodel
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
pwd_ = os.path.dirname(os.path.abspath(__file__))
dataDir = pwd_ + "/data/"
np.random.seed(123)


def main():
    # load the iris dataset
    iris = np.loadtxt(dataDir + "iris.csv", skiprows=1, usecols=(1,2,3,4,6), delimiter=',')
    # class response
    y = iris[:, -1].astype(int)
    # explanatory vars
    x = iris[:, 0:-1]
    # fit boosted classification tree to data
    iris_gbt = GBCTmodel(maxTreeDepth=2, learning_rate=0.2, subsample=0.6)
    iris_gbt.train(x[:, 0:2], y, maxIterations=50)

    # predict
    y_hat = iris_gbt.predict(x[:, 0:2])
    print(np.array((y_hat, y)))

    plot_colors = ("b", "g", "firebrick")
    class_names = ("setosa", "versicol", "virginica")
    plt.subplot(111)
    x_min, x_max = x[:, 0].min(), x[:, 0].max()
    y_min, y_max = x[:, 1].min(), x[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 80),
                         np.linspace(y_min, y_max, 80),
                         )
    # explan = np.c_[xx.ravel(), yy.ravel(), zz.ravel(), tt.ravel()]
    explan = np.c_[xx.ravel(), yy.ravel()]
    Z = iris_gbt.predict(explan)

    # class probabilities
    # Zpb = iris_gbt.predictClassProbs(explan)
    # Z = Zpb[:,0]

    # feature importance
    feature_imp = iris_gbt.feature_importances
    print("Feature Importances:")
    print(feature_imp)
    # expected feature importances
    # [('sepal length (cm)', 0.13356069065846765),
    #  ('sepal width (cm)', 0.04486948688226873),
    #  ('petal length (cm)', 0.37067096905488794),
    #  ('petal width (cm)', 0.45089885340437574)]

    # plot predictions
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=sns.diverging_palette(220, 20, sep=20, as_cmap=True))

    # Plot the training points
    for i, n, c in zip(range(3), class_names, plot_colors):
        idx = np.where(y == i)
        plt.scatter(x[idx, 0], x[idx, 1],
                    c=c, cmap=plt.cm.Paired,
                    label="Class %s" % n, s=10)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Decision Boundary')

    plt.savefig("iris_boosted_classify_ex.png")
    plt.close()

def test_importances():
    # load the iris dataset
    iris = np.loadtxt(dataDir + "iris.csv", skiprows=1, usecols=(1,2,3,4,6), delimiter=',')
    # class response
    y = iris[:, -1].astype(int)
    # explanatory vars
    x = iris[:, 0:-1]
    # fit boosted classification tree to data
    iris_gbt = GBCTmodel(maxTreeDepth=3, learning_rate=0.2, subsample=0.7)
    iris_gbt.train(x, y, maxIterations=101)

    # feature importance
    feature_imp = iris_gbt.feature_importances
    print("Feature Importances:")
    print("sepal_len, sepal_width, petal_len, petal_width")
    print(feature_imp)
    # expected feature importances
    # [('sepal length (cm)', 0.13356069065846765),
    #  ('sepal width (cm)', 0.04486948688226873),
    #  ('petal length (cm)', 0.37067096905488794),
    #  ('petal width (cm)', 0.45089885340437574)]

    # test against scikit learn


if __name__ == "__main__":
    main()
    test_importances()
