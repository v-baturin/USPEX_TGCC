'''
@file        AtomisticStage_Test.py
@author:     Vladimir Baturin
@copyright:  2023 Oganov's Lab. All rights reserved.
@contact:    vsbat@yandex.ru
@date        13 June 2013
@brief       Class for testing AtomisticStage.
'''

import unittest
from pathlib import Path

import numpy as np

from ..AtomisticStage import AtomisticStage
from ...components import AtomisticRepresentation, AtomisticPoolEntry
from ase.geometry import get_distances

PATH_WITH_TESTS = Path(__file__).parent

class AtomisticStage_Test(unittest.TestCase):
    '''

    '''
    @staticmethod
    def checkWrapped(source, sink):
        mol_no = 0
        for i, molSource in enumerate(source.system['molecules']):
            if len(molSource) > 1:
                print(f"\nMolecule {mol_no}")
                mol_no += 1
                molSink = sink.system['molecules'][i]
                distMatSource = {}
                distMatSink = {}
                diff = {}
                for pbc in ((0, 0, 0), (1, 1, 1)):
                    distMatSource[pbc[0]] = \
                    get_distances(molSource.getCartesianCoordinates(), cell=source.system['cell'].getCellVectors(),
                                  pbc=pbc)[1]

                    distMatSink[pbc[0]] = \
                    get_distances(molSink.getCartesianCoordinates(), cell=sink.system['cell'].getCellVectors(),
                                  pbc=pbc)[1]
                    diff[pbc[0]] = np.max(np.abs(distMatSource[pbc[0]] - distMatSink[pbc[0]]))
                    print(f'{pbc[0]}D-dist = ', diff[pbc[0]])
                if np.abs(diff[0] - diff[1]) > 0.1:
                    return True
        return False


    def test_fixMoleculesWrapping(self):
        badWrappingFilePath = PATH_WITH_TESTS/'mol_wrapping_POSCARS.uspex'
        systemSource, systemSink = AtomisticRepresentation.readAtomicStructures(badWrappingFilePath)
        source = AtomisticPoolEntry(**systemSource)
        sink = AtomisticPoolEntry(**systemSink)
        self.assertTrue(self.checkWrapped(source, sink))
        AtomisticStage.fixMoleculesWrapping(source, sink)
        self.assertRaises(KeyError, sink.getProperty, 'isBad')
        self.assertFalse(self.checkWrapped(source, sink))
        brokenMolFilePath = PATH_WITH_TESTS / 'mol_wrapping_POSCARS_brokenMol.uspex'
        systemSource, systemSink = AtomisticRepresentation.readAtomicStructures(brokenMolFilePath)
        systemSource['ID'] = 0
        systemSink['ID'] = 1
        source = AtomisticPoolEntry(**systemSource)
        sink = AtomisticPoolEntry(**systemSink)
        AtomisticStage.fixMoleculesWrapping(source, sink)
        self.assertTrue(sink.getProperty('isBad'))




