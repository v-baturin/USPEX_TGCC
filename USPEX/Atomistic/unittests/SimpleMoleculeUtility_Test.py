import unittest

from pathlib import Path

from ...Optimizers.PoolEntry import PoolEntry, EntryFlavour
from ...components import SimpleMoleculeUtility, Atomistic, AtomicStructureRepresentation, BondUtility

PATH_WITH_TESTS = Path(__file__).parent


class SimpleMoleculeUtility_Test(unittest.TestCase):

    def setUp(self) -> None:
        PoolEntry.createEngine(":memory:")

    def test_checkMinDistances(self):
        atomistic = Atomistic()
        extensions = {'atomistic': atomistic.propertyExtension(atomistic)}
        badDistFilePath = PATH_WITH_TESTS / "badIntermolDist_POSCAR.uspex"
        molPath = PATH_WITH_TESTS / "AlH6.xyz"
        badDistSys = Atomistic.readAtomicStructures(badDistFilePath)
        molecules = {'mol_alh6': AtomicStructureRepresentation.readXYZ(molPath.as_posix())}
        custom_iondist = {'H H': 1.9}
        bondUtility = BondUtility(volumeType=0, ionDistances=custom_iondist, cutoff='vdw')
        simpleMoleculeUtility = SimpleMoleculeUtility(molecules)
        system = PoolEntry.newEntry(
            EntryFlavour(extensions=extensions, **{'.howCome': 'Seeds', '.parent': None}, **badDistSys[0]))
        structure = system.getProperty('structure', extension='atomistic')
        minDistMatrix = bondUtility.getDistances(structure.getAtomTypes(), 0)
        self.assertTrue(simpleMoleculeUtility.checkMinDistances(system, minDistMatrix))



