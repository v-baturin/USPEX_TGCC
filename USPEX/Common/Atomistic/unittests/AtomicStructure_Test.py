'''
@file        ConvexHullTest.py
@author:     Pavel Bushlanov
@copyright:  2019 Oganov's Lab. All rights reserved.
@contact:    paulbush@mail.ru
@date        17 November 2019
@brief       Class for AtomicStructure testing
'''

import unittest
import os

from ..Crystal import Crystal


PREFIX = os.path.dirname(os.path.abspath(__file__))


class AtomicStructure_Test(unittest.TestCase):

    def test_isMoleculesDistinct1(self):
        with open("{}/CNHO_1_system".format(PREFIX), "rt") as f:
            system = Crystal.fromJSON(f.read())
        self.assertFalse(system.isMoleculesDistinct())

    def test_isMoleculesDistinct2(self):
        with open("{}/CNHO_2_system".format(PREFIX), "rt") as f:
            system = Crystal.fromJSON(f.read())
        self.assertTrue(system.isMoleculesDistinct())
