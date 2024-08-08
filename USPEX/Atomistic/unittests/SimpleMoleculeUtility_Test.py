import unittest

from pathlib import Path

from ...DataModel.Engine import Engine
from ...DataModel.Flavour import Flavour
from ...DataModel.Entry import Entry
from ...components import SimpleMoleculeUtility, Atomistic, AtomicStructureRepresentation, BondUtility

PATH_WITH_TESTS = Path(__file__).parent


class SimpleMoleculeUtility_Test(unittest.TestCase):

    def setUp(self) -> None:
        Engine.createEngine(":memory:")

    def test_checkMinDistances(self):
        atomistic = Atomistic()
        extensions = dict(
            atomistic = (atomistic, atomistic.propertyExtension.propertyTable),
        )
        badDistFilePath = PATH_WITH_TESTS / "badMolDist_POSCAR.uspex"
        molPath = PATH_WITH_TESTS / "AlH2.xyz"
        badDistSys = Atomistic.readAtomicStructures(badDistFilePath)
        molecules = {'mol_alh6': AtomicStructureRepresentation.readXYZ(molPath.as_posix())}
        custom_iondist = {'H H': 1.9}
        bondUtility = BondUtility(volumeType=0, ionDistances=custom_iondist, cutoff='vdw')
        simpleMoleculeUtility = SimpleMoleculeUtility(molecules)
        system = Entry.newEntry(
            Flavour(extensions=extensions, **{'.howCome': 'Seeds', '.parent': None}, **badDistSys[0]))
        structure = system.getProperty('structure', extension='atomistic')
        minDistMatrix = bondUtility.getDistances(structure.getAtomTypes(), 0)
        self.assertFalse(simpleMoleculeUtility.checkMinDistances(system, minDistMatrix))



