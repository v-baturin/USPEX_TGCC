import os
import unittest
from time import time
import numpy as np
from collections import Counter
from os.path import join as pj

from ....components import Cell, CellUtility, SimpleMoleculeUtility, BondUtility, Element, Conditions,\
    AtomicDisassembler, AtomisticRepresentation
from USPEX.Atomistic.VolumeEstimator import VolumeEstimator
from USPEX.Atomistic.Operators.symope.symope_cluster import symope_cluster
from USPEX.Atomistic.Operators.RandSymClusters import determineOperations


class RandSymClusters_Test(unittest.TestCase):

    def setUp(self) -> None:
        self.test_pbc = (0, 0, 0)
        self.nsym = ('E C2 D2 C4 C3 C6 T S2 Ch1 Cv2 S4 S6 Ch3 Th Ch2 Dh2 Ch4 D3 Ch6 O D4 Cv3 D6 Td Cv4 Dd3 Cv6 Oh ' +
                     'Dd2 Dh3 Dh4 Dh6 Oh C5 S5 S10 Cv5 Ch5 D5 Dd5 Dh5 I Ih').split()
        self.volumeType = 0
        self.MAX_RANDOM_FAILED_DIST = 10000
        self.MAX_RANDOM_TIME = np.inf
        self.MAX_ATTEMPTS = 150
        self.MINAT = 35
        self.MAXAT = 120
        self.ATTEMPTS_ROTATION = 1
        self.cellUtility = CellUtility(dim=0)
        self.simpleMoleculeUtility = SimpleMoleculeUtility()
        BondUtility.registerTypes(Element, AtomicDisassembler)
        self.bondUtility = BondUtility(volumeType=self.volumeType)
        self.conditions = Conditions()

    def test_symope_cluster(self, outfolder=None):
        if outfolder is not None:
            os.makedirs(outfolder, exist_ok=True)
        centerMinDistMatrix = np.array([[1.09800171]])
        symbols = ['Mo']
        successN = 0
        for sym in self.nsym:
            n_at = self.MINAT
            success = False
            tries = 0
            while not success and tries <= self.MAX_ATTEMPTS and n_at <= self.MAXAT:
                tries += 1
                numIons = [n_at]
                elementalComposition = Counter({Element('Mo'): n_at})
                badSymmetryCounter = 0
                startTime = time()
                failedDist = 0
                distCoeff = 1.
                while True:
                    endTime = time()
                    failedTime = endTime - startTime
                    if failedDist > self.MAX_RANDOM_FAILED_DIST or failedTime > self.MAX_RANDOM_TIME:
                        if distCoeff > 0.8:
                            if failedTime > self.MAX_RANDOM_TIME:
                                print(
                                    f'WARNING! Can not generate a structure after {self.MAX_RANDOM_TIME / 60} minutes. '
                                    'The minimum distance threshold will be lowered by 10%.\n')
                            else:
                                print(
                                    f'WARNING! Can not generate a structure after {self.MAX_RANDOM_FAILED_DIST} tries. '
                                    'The minimum distance threshold will be lowered by 10%.\n')
                            failedDist = 0
                            startTime = time()
                            distCoeff *= 0.9

                        else:
                            msg = f'Could not generate a structure after {self.MAX_RANDOM_FAILED_DIST} tries or {self.MAX_RANDOM_TIME / 60} minutes.\n'
                            msg += 'Please check the input files. The calculation has to stop.\n'
                            msg += 'Possible reasons: unreasonably big IonDistances.\n'
                            msg += 'Remember they should be much smaller than the real interatomic distances,\n'
                            msg += 'but not too small for pseudopotential overlap errors to kill interatomic repulsion.\n'
                            print(msg)
                            raise RuntimeError("RandSym_clusters failed.")

                    if badSymmetryCounter > self.MAX_ATTEMPTS:
                        print(f"Failed to generate cluster of {n_at} atoms with {sym} symmetry after {self.MAX_ATTEMPTS} tries")
                        n_at += 1
                        break
                    else:
                        badSymmetryCounter += 1

                    try:
                        estimatedVolume = 1.5 * VolumeEstimator(self.volumeType).calcCompositionVolume(elementalComposition, 0.0)

                        randcell = RandSymClusters_Test.rand_orthog_lattice(estimatedVolume)
                        candidate, lat = symope_cluster(distCoeff * centerMinDistMatrix, sym,
                                                        [n_at], randcell)
                        self.assertEqual(numIons[0], len(candidate))
                        name, cell, operations = determineOperations(lat, numIons, candidate)
                        operations = dict(zip(symbols, operations))
                        cell = self.cellUtility.adjustCell(cell, estimatedVolume, sum(numIons), baseCell=None)
                        offspring = self.simpleMoleculeUtility.populateStructure(cell, operations)
                        atomSymbols, atomDistances, disassembler = self.simpleMoleculeUtility.getMinDistances(
                            **offspring)
                        minDistMatrix = self.bondUtility.getDistances(atomSymbols, pressure=0.0)
                        if np.all(atomDistances >= distCoeff * minDistMatrix):
                            self.conditions.putConditions(offspring)
                            structure, disassembler = self.simpleMoleculeUtility.atomicDisassemblerType.assemble(
                                **offspring)
                            if self.bondUtility.isConnected(structure):
                                # structure
                                print(f"Structure with {n_at} attoms generated with symmetry {sym}")
                                successN += 1
                                success = True
                                if outfolder is not None:
                                    AtomisticRepresentation.writePOSCAR(pj(outfolder, f'POSCAR_{sym}'), structure, label='EA1')
                                break
                            else:
                                raise RuntimeError('non-connected structure')
                        else:
                            raise RuntimeError('Distances too short')

                    except Exception as e:
                        if 'Impossible to build ' in str(e):
                            n_at += 1

                    failedDist += 1
        self.assertEqual(successN, len(self.nsym))



    @staticmethod
    def rand_orthog_lattice(volume):
        randcell = np.random.random(3)
        randcell *= (volume / np.prod(randcell)) ** (1 / 3)
        return Cell.initFromCellVectors((1, 1, 1), np.diag(randcell))
