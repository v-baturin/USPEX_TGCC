'''
@file        optLatticeTest.py
@author:     Vladimir Baturin
@copyright:  2018 Oganov's Lab. All rights reserved.
@contact:    v.baturin@skoltech.ru
@date_start  08 November 2018
@brief       Class for optLattice testing
'''


import unittest
import numpy as np

import ..optLattice


class optLatticeTest(unittest.TestCase):

    def test_optLattice(self):
        v1 = [0.1, 0.2, 0.3]
        v2 = [0.3, 0.2, 0.4]
