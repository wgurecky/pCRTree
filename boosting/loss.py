#!/usr/bin/python3
##!
# \brief Loss functions
# \date Feb 4 2017
# \author William Gurecky
##
import abc
import re
import numpy as np


class AbastractLoss(metaclass=abc.ABCMeta):
    """!
    @brief Loss function base class
    """
    def __int__(self, *args, **kwargs):
        pass

    @property
    def params(self, params):
        return self._params

    @params.setter
    def params(self, params):
        self._params = params

    @abc.abstractmethod
    def loss(self, y, yhat):
        """!
        @brief Pure virtual loss function.
        """
        raise NotImplementedError

    def gradLoss(self, y, yhat):
        """!
        @brief Computes gradient of loss function.
        The default implementation is to compute numerical
        derivative.  If possible, should override

        Jacobian of loss function:
        \f[
        -\nabla L(y, \hat y) = \frac{\partial \sum_i{(y_i - f(x_i)})^2}{\partial f(x_i)}
        \f]
        Where \f$ f(x_i) \f$ is the current gbt model predicting \f$\hat y_i\f$.
        @param y  (1d_array) Input traning y data vector
        @param yhat (1d_array) model predicted vector
        """
        import numdifftools as nd
        lambda_loss = lambda yp: self.loss(y, yp)
        return -np.diagonal(nd.Gradient(lambda_loss)(yhat))

    def sumLoss(self, y, yhat):
        """!
        """
        return np.sum(self.loss(y, yhat))

    def sse(self, y, yhat):
        """!
        @brief Sum squared errors.
        """
        return np.sum((y - yhat) ** 2)

    def var(self, y, yhat):
        """!
        @brief Sum squared errors.
        """
        return np.sum((y - yhat) ** 2) / len(y)


# =========================================================================== #
# Concrete Classes
# =========================================================================== #
class SquaredLoss(AbastractLoss):
    """!
    @brief squared error loss function concrete class.
    """
    def __init__(self, *args, **kwargs):
        self.name = kwargs.pop("name", "se")
        super(AbastractLoss, self).__init__(*args, **kwargs)

    def loss(self, y, yhat):
        return (y - yhat) ** 2.

    def gradLoss(self, y, yhat):
        return (y - yhat)


class HuberLoss(AbastractLoss):
    """!
    @brief Absolute error loss function concrete class.
    """
    def __init__(self, *args, **kwargs):
        self.name = kwargs.pop("name", "abs")
        super(AbastractLoss, self).__init__(*args, *kwargs)

    def loss(self, y, yhat):
        delta = 1.0
        return delta ** 2. * (np.sqrt(1. + ((y - yhat) / delta) ** 2.)) - 1.


# =========================================================================== #
# Factory
# =========================================================================== #
class FLoss(object):
    """!
    @Sbrief Simple factory which returns a concrete loss class instance
    given a loss name (str).
    """
    def __new__(cls, name):
        """!
        @param name (str) name of loss function
        """
        if re.match("se", name):
            return SquaredLoss(name=name)
        if re.match("huber", name):
            return HuberLoss(name=name)
        else:
            print("WARNING: Requested loss unavalible. \
                   Falling back to squared error loss.")
            return SquaredLoss(name=name)

    def __init__(self, name):
        self.name = name


if __name__ == "__main__":
    # test setup
    a = np.array([0.01, 0.1, 0.3, 1,2,3, 4])
    b = np.array([0.02, 0.3, 1, 2,4,6, 10])
    my_loss = FLoss("se")
    print(my_loss.loss(a, b))
    print(my_loss.gradLoss(a, b))
    #
    my_hloss = FLoss("huber")
    print(my_hloss.loss(a, b))
    print(my_hloss.gradLoss(a, b))
