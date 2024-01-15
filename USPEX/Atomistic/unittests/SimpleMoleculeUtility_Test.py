import unittest

from pathlib import Path

from ...Optimizers.PoolEntry import PoolEntry, EntryFlavour
from ...components import SimpleMoleculeUtility, Atomistic

PATH_WITH_TESTS = Path(__file__).parent


class SimpleMoleculeUtility_Test(unittest.TestCase):

    def setUp(self) -> None:
        PoolEntry.createEngine(":memory:")
        self.utility = SimpleMoleculeUtility()

    def test_checkMinDistances(self):
        atomistic = Atomistic()
        extensions = {'atomistic': atomistic.propertyExtension(atomistic)}
        badDistFilePath = PATH_WITH_TESTS / "badIntermolDist_POSCAR.uspex"
        molPath = PATH_WITH_TESTS
        badDistSys = Atomistic.readAtomicStructures(badDistFilePath)
        molecules = {'mol_alh6': }
        system = PoolEntry.newEntry(
            EntryFlavour(extensions=extensions, **{'.howCome': 'Seeds', '.parent': None}, **badDistSys))

