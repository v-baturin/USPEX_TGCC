"""
USPEX.Common.Atomistic.Fingerprints.unittests.Fingerprints_Test
===============================================================

Class for Fingerprints testing

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import unittest
import os
from ase.io import read

from ...AtomicStructure import AtomicStructure
from ..cosine_distance import cosine_distance


PATH_WITH_TESTS = os.path.dirname(os.path.abspath(__file__))


class Fingerprints_Test(unittest.TestCase):

    def test_atomic(self):
        system1 = read(PATH_WITH_TESTS + '/system1_POSCAR')
        system1 = AtomicStructure(symbols = system1.get_chemical_symbols(),
                                  positions = system1.get_positions(),
                                  cell = system1.get_cell())
        system2 = read(PATH_WITH_TESTS + '/system2_POSCAR')
        system2 = AtomicStructure(symbols = system2.get_chemical_symbols(),
                                  positions = system2.get_positions(),
                                  cell = system2.get_cell())
        fp1 = system1.fingerprint
        fp2 = system2.fingerprint
        system1.fingerprintTolerance = 1.0e-6
        self.assertTrue(system1 == system2)

# import numpy as np
# np.set_printoptions(threshold=10000, precision=4, suppress=True)
# print(f1.fingerprint[3])
# print(f2.fingerprint[0])
# dm1 = make_matrices(system1)
# dm2 = make_matrices(system2)
# btype1 = dm1[:, 1] * 2 + dm1[:, 2]
# btype2 = dm2[:, 1] * 1 + dm2[:, 2]
# print(len(dm1[np.where(btype1==3), 3][0]), len(dm2[:, 3]))
