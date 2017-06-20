"""!
@brief Boosted boston houseing price example
comapre with results from:
    http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html
"""
from sklearn import datasets
from boosting.gbm import GBRTmodel
import numpy as np
from sklearn.utils import shuffle


def main():
    # load data
    boston = datasets.load_boston()
    X, y = shuffle(boston.data, boston.target, random_state=13)
    X = X.astype(np.float32)
    offset = int(X.shape[0] * 0.9)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]
    
    # fit model
    iters = 100
    gbt = GBRTmodel(maxTreeDepth=4, learning_rate=0.1, subsample=0.5)
    gbt.train(X_train, y_train, maxIterations=iters)

    # print importances
    print("Feature Importances")
    print(gbt.feature_importances)



if __name__ == "__main__":
    main()
