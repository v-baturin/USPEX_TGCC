'''
@file        AtomisticStage_Test.py
@author:     Vladimir Baturin
@copyright:  2023 Oganov's Lab. All rights reserved.
@contact:    vsbat@yandex.ru
@date        13 June 2013
@brief       Class for testing AtomisticStage.
'''

import unittest
import numpy as np
from pathlib import Path
from ase.geometry import get_distances

from ..AtomisticStage import AtomisticStage
from ...DataModel.Engine import Engine
from ...DataModel.Flavour import Flavour
from ...DataModel.Entry import Entry
from ...components import Atomistic

PATH_WITH_TESTS = Path(__file__).parent

class FakeSimpleMoleculeUtility:
    whatToCheckInMolecules = 'all'
    integrityTol = 0.1
class FakeUtilities:
    simpleMoleculeUtility = FakeSimpleMoleculeUtility()
class FakeTarget:
    utilities = FakeUtilities()

class AtomisticStage_Test(unittest.TestCase):
    '''
    '''

    def setUp(self) -> None:
        Engine.createEngine(":memory:")

    @staticmethod
    def checkWrapped(system):
        mol_no = 0
        cell = system.getProperty('cell', extension='atomistic', suffix='origin')
        for i, molSource in enumerate(system.getProperty('molecules', extension='atomistic', suffix='origin')):
            if len(molSource) > 1:
                # print(f"\nMolecule {mol_no}")
                mol_no += 1
                molSink = system.getProperty('molecules', extension='atomistic', suffix='0')[i]
                distMatSource = {}
                distMatSink = {}
                diff = {}
                for pbc in ((0, 0, 0), (1, 1, 1)):
                    distMatSource[pbc[0]] = \
                    get_distances(molSource.getCartesianCoordinates(), cell=cell.getCellVectors(),
                                  pbc=pbc)[1]

                    distMatSink[pbc[0]] = \
                    get_distances(molSink.getCartesianCoordinates(), cell=cell.getCellVectors(),
                                  pbc=pbc)[1]
                    diff[pbc[0]] = np.max(np.abs(distMatSource[pbc[0]] - distMatSink[pbc[0]]))
                    # print(f'{pbc[0]}D-dist = ', diff[pbc[0]])
                if np.abs(diff[0] - diff[1]) > 0.1:
                    return True
        return False


    def test_fixMoleculesWrapping(self):
        AtomisticStage.registerTypes(lambda *args, **kwargs: None)
        atomisticStage = AtomisticStage(tag='0',
                                        source='origin',
                                        perturbate=False,
                                        target=FakeTarget(),
                                        environmentStyle=None,
                                        vacuumSize=0)
        atomistic = Atomistic()
        extensions = {'atomistic': atomistic.propertyExtension()}
        for badWrappingFilePath in [PATH_WITH_TESTS / 'dewrapping_POSCAR.uspex',
                                    PATH_WITH_TESTS/'mol_wrapping_POSCARS.uspex']:
            systemSource, systemSink = Atomistic.readAtomicStructures(badWrappingFilePath)
            system = Entry.newEntry(Flavour(extensions=extensions, **{'.howCome': 'Seeds', '.parent': None}, **systemSource))
            system.addFlavour('0', Flavour(extensions=extensions, **systemSink))
            self.assertTrue(self.checkWrapped(system))
            atomisticStage.checkAndFixMolecules(system)
            self.assertFalse(self.checkWrapped(system))
        brokenMolFilePath = PATH_WITH_TESTS / 'mol_wrapping_POSCARS_brokenMol.uspex'
        systemSource, systemSink = Atomistic.readAtomicStructures(brokenMolFilePath)
        system = Entry.newEntry(Flavour(extensions=extensions, **{'.howCome': 'Seeds', '.parent': None}, **systemSource))
        system.addFlavour('0', Flavour(extensions=extensions, **systemSink))
        atomisticStage.checkAndFixMolecules(system)
        self.assertTrue(system.getProperty('isBad', suffix='0'))




