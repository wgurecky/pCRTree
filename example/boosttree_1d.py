#!/usr/bin/python3
from boosting.gbm import GBRTmodel
from scipy.interpolate import griddata
from matplotlib import gridspec
import seaborn as sns
import pandas as pd
import os
import scipy
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
MPL = False
try:
    from pylab import cm
    import matplotlib.pyplot as plt
    MPL = True
except: pass


def example_qunatile_reg_loop(q_tile=0.9, n_loop=2):
        np.random.seed(1)
        def f(x):
            heaviside = np.heaviside(x - 5.0, 1.0) * 12.
            return x * np.sin(x) + heaviside + 10.

        boost_settings = {
                'n_estimators': 2000,
                'max_depth': 3,
                'subsample': 0.7,
                'learning_rate': 0.08,
                'loss': 'quantile',
                'verbose': True
                }

        q_predict_list, q_residual_list = [], []
        sk_q_predict_list, sk_q_residual_list = [], []
        fig = plt.figure()
        for i in range(n_loop):
            boost_settings['alpha'] = q_tile
            sklearn_reg = GradientBoostingRegressor(**boost_settings)

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
            gbt = GBRTmodel(max_depth=1, learning_rate=0.08, subsample=0.7, loss="quantile", alpha=q_tile)
            gbt.train(X, y, n_estimators=2000)
            sklearn_reg.fit(X, y)
            q_tile_hat = gbt.predict(xx)
            sk_q_tile_hat = sklearn_reg.predict(xx)

            # truth
            q_tile_true = f(xx) + normal_q_tile

            # Plot the function, the prediction and the 90% confidence interval based on
            # the MSE
            plt.plot(xx, q_tile_hat, 'r-', alpha=0.2, lw=0.5)
            q_predict_list.append(q_tile_hat)
            q_residual_list.append(q_tile_hat.flatten() - q_tile_true.flatten())

            # sklearn results
            sk_q_predict_list.append(sk_q_tile_hat)
            sk_q_residual_list.append(sk_q_tile_hat.flatten() - q_tile_true.flatten())

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

        # compute theoredical quantile residual distribution
        qth_numerator = q_tile * (1 - q_tile)
        qth_denom = n_samples_per_edit * (norm_rv.pdf(norm_rv.ppf(q_tile))) ** 2.0
        std_dev_q_theory = np.sqrt(qth_numerator / qth_denom)

        # flattened view of quantile predictions for violin residual plots
        flat_q = np.array(q_residual_list).flatten()
        sk_flat_q = np.array(sk_q_residual_list).flatten()

        # draw results from theory distribution
        th_flat_q = np.random.normal(0, std_dev_q_theory, size=np.size(flat_q))

        with sns.axes_style("whitegrid", {"grid.linestyle": '--', "grid.alpha": 0.6}):
            sns.set_palette(sns.diverging_palette(250, 15, s=70, l=45, n=3, center="light"))
            sns.set_color_codes()
            fig = plt.figure(figsize=(8, 6), dpi=120)
            gs = gridspec.GridSpec(1, 3)
            ax1 = fig.add_subplot(gs[:2])
            ax1.ticklabel_format(style='sci', scilimits=(0,0), useMathText=True, axis='y')
            ax2 = fig.add_subplot(gs[2], sharey=ax1)
            plt.setp(ax2.get_yticklabels(), visible=False)
            for q_resid, sk_q_resid, in zip(q_residual_list, sk_q_residual_list):
                ax1.plot(xx, q_resid, 'b', ls=':', alpha=0.2)
                ax1.plot(xx, sk_q_resid, 'r', ls=':', alpha=0.2)
            ax1.plot(xx, np.average(q_residual_list, axis=0), 'b', lw=1, label='gBRTree')
            ax1.plot(xx, np.average(sk_q_residual_list, axis=0), 'r', lw=1, label='sk-learn')
            ax1.axhline(0.0, c='k', lw=0.5)
            ax1.set_xlabel('$x$')
            ax1.set_ylabel(r"Residual $\hat q_{" + str(q_tile) + "} - q_{" + str(q_tile) + "}[f(x)] $")

            # create labels for violin plot
            mc_tag_list = ['pCRTree' for item in flat_q]
            imp_tag_list = ['sk-learn' for item in sk_flat_q]
            th_tag_list = ['Theory' for item in sk_flat_q]
            bmass_sample_dframe = pd.DataFrame.from_dict({"": np.concatenate((flat_q, th_flat_q, sk_flat_q, )),
                                                          'Method': np.array(mc_tag_list + th_tag_list + imp_tag_list )})
            sns.violinplot(x="Method", y="", data=bmass_sample_dframe, split=True, ax=ax2)
            ax1.set_title(r"$SE[q_{" + str(q_tile) + "}]_{th}$ = %0.3f " % (std_dev_q_theory) + \
                          "\n" + \
                          r"$SE[q_{" + str(q_tile) + "}]_{pCRTree}$ = %0.3f" % (np.std(flat_q)) + \
                          "\n" + \
                          r"$SE[q_{" + str(q_tile) + "}]_{sk-learn}$ = %0.3f" % (np.std(sk_flat_q)), fontsize=10)
            ax1.legend()
            plt.tight_layout()
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
