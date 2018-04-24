"""!
@brief Boosted boston houseing price example
comapre with results from:
    http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html
"""
from sklearn import datasets
from boosting.gbm import GBRTmodel
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def main():
    # load data
    boston = datasets.load_boston()
    X, y = shuffle(boston.data, boston.target, random_state=13)
    X = X.astype(np.float32)
    offset = int(X.shape[0] * 0.9)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]
    # fit model
    iters = 500
    gbt = GBRTmodel(max_depth=4, learning_rate=0.02, subsample=0.6, scale=False)
    gbt.train(X_train, y_train, n_estimators=iters)

    # print importances
    print("Feature Importances")
    feature_importance = gbt.feature_importances_
    print(feature_importance)

    # sklearn predict
    try:
        from sklearn.ensemble import GradientBoostingRegressor as SkGbt
        sk_gbt = SkGbt(max_depth=4, loss='ls', subsample=0.9, learning_rate=0.01, n_estimators=500)
        sk_gbt.fit(X_train, y_train)
        sk_feature_importance = sk_gbt.feature_importances_
        print("Scikit-learn Feature Importances")
        print(sk_feature_importance)
        #
        sorted_idx = np.argsort(sk_feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.subplot(1, 1, 1)
        plt.barh(pos, sk_feature_importance[sorted_idx], align='center')
        plt.yticks(pos, boston.feature_names[sorted_idx])
        plt.xlabel('Fractional Importance')
        plt.title('Variable Importance')
        plt.savefig("boston_feature_imp_sklearn.png")
        plt.close()
    except:
        pass

    # do not require plotting to succeed
    try:
        # plot normed feature importances
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.subplot(1, 1, 1)
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, boston.feature_names[sorted_idx])
        plt.xlabel('Fractional Importance')
        plt.title('Variable Importance')
        plt.savefig("boston_feature_imp.png")
        plt.close()
    except:
        pass


if __name__ == "__main__":
    main()
