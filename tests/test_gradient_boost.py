#!/usr/bin/python3
from boosting.gbm import GBRTmodel
import unittest
import os
import numpy as np
pwd_ = os.path.dirname(os.path.abspath(__file__))
dataDir = pwd_ + "/data/"
np.random.seed(123)


class TestGradBoosting(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    def test1dBoostedReg(self):
        pass

    def test2dBoostedReg(self):
        pass
