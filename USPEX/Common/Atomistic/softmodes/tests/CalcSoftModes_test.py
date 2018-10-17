import pickle as pcl
import unittest

from lib.Atomistic.softmodes.calcSoftModes import calcSoftModes


class CalcSoftModes_test(unittest.TestCase):
    def test_system1(self):
        with open('system', 'rb') as f:
            system = pcl.load(f)
        R_val = [1.41, 1.21, 0.66]
        N_val = [2, 3, 6]
        val = [2, 3, 2]
        calcSoftModes(system, R_val, N_val, val)
