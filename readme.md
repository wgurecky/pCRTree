[![Build Status](https://travis-ci.org/wgurecky/pCRTree.svg?branch=master)](https://travis-ci.org/wgurecky/pCRTree)

About
======

This package implements gradient boosted classification and regression trees.
In addition to piecewise constant base learners, piecewise linear regression trees are also implemented.

Gradient boosting is a supervised learning technique applicable to nonlinear regression and classification problems.

pCRTree is split into two subpackages:

- dtree: Classification and regression trees.
- boosting: Gradient boosting methods.


Examples
========

Supports N-Dimensional regression and classification.

Regression
---

Gradient boosted piecewise linear regression trees:

Example in `example/boost_lin_tree_1d.py`

Gradient boosted traditional regressing trees:

Quantile regression:

Classification
---

Example in `example/classify_dblgauss.py`


Install
========

Depends:

- numpy
- scipy
- numba

Optional:

- matplotlib
- pytest (for testing)
- sklearn (for example and test benchmarks)

For developers:

    $python3 setup.py develop --user

Users:

    $python3 setup.py install --user

