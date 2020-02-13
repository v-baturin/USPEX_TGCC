import os
import pickle as pcl
import unittest

from ..calcSoftModes import calcSoftModes


class CalcSoftModes_test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    def test_system1(self):
        with open('{}/system'.format(self.CURRENT_DIR), 'rb') as f:
            system = pcl.load(f)
        R_val = [1.41, 1.21, 0.66]
        N_val = [2, 3, 6]
        val = [2, 3, 2]
        calcSoftModes(system, R_val, N_val, val)
