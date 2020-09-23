#!/usr/bin/python3
from boosting.gbm import GBRTmodel
from matplotlib import gridspec
import numpy as np
MPL = False
try:
    from pylab import cm
    import matplotlib.pyplot as plt
    MPL = True
except: pass


def example_boosed_lin_reg_lin(f, plot_name='1d_boosted_regression_lin_lin_ex.png'):

        n_samples_per_edit = 1
        X = np.atleast_2d(np.linspace(0, 10.0, 80).repeat(n_samples_per_edit)).T
        X = X.astype(np.float32)
        y = f(X).ravel()

        std_dev = 4.2
        noise = np.random.normal(0, std_dev, size=y.shape)
        y += noise
        y = y.astype(np.float32)

        # Mesh the input space for evaluations of the real function, the prediction and
        # its MSE
        xx = np.atleast_2d(np.linspace(0, 10, 1000)).T
        xx = xx.astype(np.float32)

        # fit to median
        gbt = GBRTmodel(max_depth=1, learning_rate=0.03, subsample=0.8, loss="se", tree_method="lin", minSplitPts=10, minDataLeaf=20)
        gbt.train(X, y, n_estimators=400)
        y_median = gbt.predict(xx)

        if MPL:
            # Plot the function, the prediction and the 90% confidence interval based on
            # the MSE
            fig = plt.figure()
            plt.plot(X, y, 'b.', markersize=2, label=u'Observations', alpha=0.3)
            plt.plot(xx, y_median, 'r-', label=r'$\hat \mu$')
            plt.plot(xx, f(xx), 'g', label=u'$f(x) = 0.25x+10$')
            plt.xlabel('$x$')
            plt.ylabel('$f(x)$')
            # plt.ylim(-10, 20)
            plt.legend(loc='upper left')
            plt.savefig('lin_' + plot_name + '.png', dpi=120)
            plt.close()

        # fit to median
        gbt = GBRTmodel(max_depth=1, learning_rate=0.03, subsample=0.8, loss="se", tree_method="cart", minSplitPts=4)
        gbt.train(X, y, n_estimators=400)
        y_median = gbt.predict(xx)

        if MPL:
            # Plot the function, the prediction and the 90% confidence interval based on
            # the MSE
            fig = plt.figure()
            plt.plot(X, y, 'b.', markersize=2, label=u'Observations', alpha=0.3)
            plt.plot(xx, y_median, 'r-', label=r'$\hat \mu$')
            plt.plot(xx, f(xx), 'g', label=u'$f(x) = 0.25x+10$')
            plt.xlabel('$x$')
            plt.ylabel('$f(x)$')
            # plt.ylim(-10, 20)
            plt.legend(loc='upper left')
            plt.savefig('const_' + plot_name + '.png', dpi=120)
            plt.close()


def example_boosed_lin_reg_sin():
        def f(x):
            heavyside = np.heaviside(x - 5.0, 1.0) * 12.
            return x * np.sin(x) + heavyside + 10.

        n_samples_per_edit = 1
        X = np.atleast_2d(np.linspace(0, 10.0, 220).repeat(n_samples_per_edit)).T
        X = X.astype(np.float32)
        y = f(X).ravel()

        # std_dev = 1.5 + 1.0 * np.random.random(y.shape)
        std_dev = 2.0
        noise = np.random.normal(0, std_dev, size=y.shape)
        y += noise
        y = y.astype(np.float32)

        # Mesh the input space for evaluations of the real function, the prediction and
        # its MSE
        xx = np.atleast_2d(np.linspace(0, 10, 1000)).T
        xx = xx.astype(np.float32)

        # fit to median
        gbt = GBRTmodel(max_depth=2, learning_rate=0.01, subsample=0.5, loss="se", tree_method="lin", minSplitPts=20, minDataLeaf=25)
        gbt.train(X, y, n_estimators=380)
        y_median = gbt.predict(xx)

        if MPL:
            # Plot the function, the prediction and the 90% confidence interval based on
            # the MSE
            fig = plt.figure()
            plt.plot(X, y, 'b.', markersize=2, label=u'Observations', alpha=0.3)
            plt.plot(xx, y_median, 'r-', label=r'$\hat q_{0.50}$')
            plt.plot(xx, f(xx), 'g', label=u'$f(x) = x\,\sin(x) + 12 H(x-5)$')
            plt.xlabel('$x$')
            plt.ylabel('$f(x)$')
            # plt.ylim(-10, 20)
            plt.legend(loc='upper left')
            plt.savefig('1d_boosted_regression_lin_sin_ex.png', dpi=120)
            plt.close()


if __name__ == "__main__":

    def f(x):
        #y = np.zeros(len(x))
        #y[np.asarray(x < 5).flatten()] = x[x < 5] * 0.25 + 10.
        #y[np.asarray(x >= 5).flatten()] = x[x >= 5] * -0.25 + 12.
        y = x ** 2.0
        return y
    example_boosed_lin_reg_lin(f, 'pos_quadratic')
    def f(x):
        #y = np.zeros(len(x))
        #y[np.asarray(x < 5).flatten()] = x[x < 5] * 0.25 + 10.
        #y[np.asarray(x >= 5).flatten()] = x[x >= 5] * -0.25 + 12.
        y = -x ** 2.0
        return y
    example_boosed_lin_reg_lin(f, 'neg_quadratic')
    example_boosed_lin_reg_sin()
