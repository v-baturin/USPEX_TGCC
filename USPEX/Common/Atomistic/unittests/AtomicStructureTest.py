'''
@file        AtomicStructureTest.py
@author:     Vladimir Baturin
@copyright:  2018 Oganov's Lab. All rights reserved.
@contact:    v.baturin@skoltech.ru
@date_start  01 November 2018
@brief       Class for AtomicStructure testing
'''


import unittest
import numpy as np

from ..AtomicStructure import AtomicStructure



class System(object):

    def __init__(self, composition, enthalpy):
        self.composition = composition
        self.enthalpy = enthalpy


class AtomicStructureTest(unittest.TestCase):

    def test_optLattice(self):
