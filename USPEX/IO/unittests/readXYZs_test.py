import unittest
import os
from os.path import join as pj

from ...components import AtomisticRepresentation


HOMEPATH = os.path.dirname(os.path.abspath(__file__))


class ReadXYZs_Test(unittest.TestCase):
    def test_read_xyzs(self):
        structures = AtomisticRepresentation.readXYZs(pj(HOMEPATH, 'several_structures.xyz'))
        self.assertEqual(len(structures), 7)
        for structure in structures:
            self.assertEqual(len(structure), 40)




