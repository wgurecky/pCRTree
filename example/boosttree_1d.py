#!/usr/bin/python3
from boosting.gbm import GBRTmodel
from scipy.interpolate import griddata
import os
import scipy
from scipy.stats import norm
import numpy as np
MPL = False
try:
    from pylab import cm
    import matplotlib.pyplot as plt
    MPL = True
except: pass


def example_qunatile_reg_loop(q_tile=0.9, n_loop=10):
        np.random.seed(1)
        def f(x):
            heavyside = np.heaviside(x - 5.0, 1.0) * 12.
            return x * np.sin(x) + heavyside + 10.

        q_predict_list, q_residual_list = [], []
        fig = plt.figure()
        for i in range(n_loop):

            n_samples_per_edit = 50
            X = np.atleast_2d(np.linspace(0, 10.0, 50).repeat(n_samples_per_edit)).T
            X = X.astype(np.float32)
            y = f(X).ravel()

            # std_dev = 1.5 + 1.0 * np.random.random(y.shape)
            std_dev = 2.0
            mu = 0.0
            noise = np.random.normal(mu, std_dev, size=y.shape)
            norm_rv = norm(0, std_dev)
            # normal_q_tile = mu + std_dev * np.sqrt(2.) * \
            #         scipy.special.erfinv(2. * norm_rv.cdf(q_tile) - 1.)
            normal_q_tile = norm_rv.ppf(q_tile)

            # apply noise to fn
            y += noise
            y = y.astype(np.float32)

            # Mesh the input space for evaluations of the real function, the prediction and
            # its MSE
            xx = np.atleast_2d(np.linspace(0, 10, 1000)).T
            xx = xx.astype(np.float32)

            # fit to quantile
            gbt = GBRTmodel(max_depth=1, learning_rate=0.1, subsample=0.8, loss="quantile", alpha=q_tile)
            gbt.train(X, y, n_estimators=1000)
            q_tile_hat = gbt.predict(xx)

            # truth
            q_tile_true = f(xx) + normal_q_tile

            # Plot the function, the prediction and the 90% confidence interval based on
            # the MSE
            plt.plot(xx, q_tile_hat, 'r-', alpha=0.2, lw=0.5)
            q_predict_list.append(q_tile_hat)
            q_residual_list.append(q_tile_hat.flatten() - q_tile_true.flatten())

        q_predict_list = np.array(q_predict_list)
        plt.plot(xx, np.average(q_predict_list, axis=0), \
                'r-', label=r'$\hat \bar q_{' + str(q_tile) + '}$', alpha=1.0, lw=1.0)
        # true fn
        plt.plot(xx, f(xx), 'g', label=u'$f(x) = x\,\sin(x) + 12 H(x-5)$')
        # true quantile
        plt.plot(xx, q_tile_true, 'b', label=u'$q_{' + str(q_tile) + '}[f(x)$]')
        # observations
        plt.plot(X, y, 'b.', markersize=2, label=u'Observations', alpha=0.3)
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        # plt.ylim(-10, 20)
        plt.legend(loc='upper left')
        plt.savefig('1d_boosted_regression_quantile_ex_' + str(q_tile) + '.png', dpi=120)
        plt.close()

        fig, ax1 = plt.subplots(1)
        for q_resid in q_residual_list:
            ax1.plot(xx, q_resid, 'r-', alpha=0.2)
        ax1.axhline(0.0, c='k', lw=0.5)
        ax1.set_xlabel('$x$')
        ax1.set_ylabel(r"Residual $\hat q_{" + str(q_tile) + "} - q_{" + str(q_tile) + "}[f(x)] $")
        plt.savefig('1d_boosted_regression_quantile_resid_' + str(q_tile) + '.png', dpi=120)
        plt.close()


def example_qunatile_reg():
        np.random.seed(1)
        def f(x):
            heavyside = np.heaviside(x - 5.0, 1.0) * 12.
            return x * np.sin(x) + heavyside + 10.

        n_samples_per_edit = 50
        X = np.atleast_2d(np.linspace(0, 10.0, 50).repeat(n_samples_per_edit)).T
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
        gbt = GBRTmodel(max_depth=1, learning_rate=0.2, subsample=0.9, loss="quantile", alpha=0.5)
        gbt.train(X, y, n_estimators=350)
        y_median = gbt.predict(xx)

        # lower
        gbt.alpha = 0.1
        gbt.train(X, y, n_estimators=350)
        y_lower = gbt.predict(xx)

        # upper
        gbt.alpha = 0.9
        gbt.train(X, y, n_estimators=350)
        y_upper = gbt.predict(xx)

        if MPL:
            # Plot the function, the prediction and the 90% confidence interval based on
            # the MSE
            fig = plt.figure()
            plt.plot(X, y, 'b.', markersize=2, label=u'Observations', alpha=0.3)
            plt.plot(xx, y_median, 'r-', label=r'$\hat q_{0.50}$')
            plt.plot(xx, y_upper, 'k-', lw=0.5, alpha=0.7)
            plt.plot(xx, y_lower, 'k-', lw=0.5, alpha=0.7)
            plt.fill(np.concatenate([xx, xx[::-1]]),
                     np.concatenate([y_upper, y_lower[::-1]]),
                     alpha=.3, fc='b', ec='None', label=r'[$\hat q_{0.10}, \hat q_{0.90}$]')
            plt.plot(xx, f(xx), 'g', label=u'$f(x) = x\,\sin(x) + 12 H(x-5)$')
            plt.xlabel('$x$')
            plt.ylabel('$f(x)$')
            # plt.ylim(-10, 20)
            plt.legend(loc='upper left')
            plt.savefig('1d_boosted_regression_quantile_ex2.png', dpi=120)
            plt.close()


if __name__ == "__main__":
    example_qunatile_reg_loop()
