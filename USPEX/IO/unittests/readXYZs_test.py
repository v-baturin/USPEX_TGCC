import unittest
from pathlib import Path

from ...components import AtomisticRepresentation


HOMEPATH = Path(__file__).parent


class ReadXYZs_Test(unittest.TestCase):
    def test_read_xyzs(self):
        structures = AtomisticRepresentation.readXYZs(HOMEPATH/'several_structures.xyz')
        self.assertEqual(len(structures), 7)
        for structure in structures:
            self.assertEqual(len(structure), 40)




