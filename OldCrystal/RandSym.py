import logging
logger = logging.getLogger(__name__)


import numpy as np
import spglib
from itertools import combinations_with_replacement

from .Random import Random
from .symope.symope_crystal import symope_crystal
from time import time

from .Crystal import Crystal
from .calcDefaultVolume import calcVolume
from ..SpaceGroups.SpaceGroups3D import Group


MAX_RANDOM_FAILED_DIST = 10000
MAX_RANDOM_TIME = 300


def determineOperations(lat, numIons, candidate):
    logger.debug("Determinig operations variants")
    cell = (lat, candidate, np.repeat(list(range(len(numIons))), numIons))
    dataset = spglib.get_symmetry_dataset(cell, symprec=1e-2)
    symbol = dataset['international']
    group = Group.getGroupFromSymbol(symbol)
    equivalent_atoms = dataset['equivalent_atoms']
    coordinates = []
    operations = []
    i = 0
    for n in numIons:
        atomCoordinates = []
        atomOperations = []
        equivalent_atoms_one_type = equivalent_atoms[i:i + n]
        for index in np.unique(equivalent_atoms_one_type):
            nodeIndices = np.where(equivalent_atoms==index)[0]
            assert len(nodeIndices) == len(np.where(equivalent_atoms_one_type==index)[0])
            atomCoordinates.append(candidate[nodeIndices])
            atomOperations.append(group.getNotPositionInvariantSubgroups(candidate[nodeIndices[0]]))
        coordinates.append(atomCoordinates)
        operations.append(atomOperations)
        i += n

    return symbol, lat, coordinates, operations


class RandSym(Random):

    logger = logger

    def __init__(self, *args, nsymN=False, nsym=None, sym_coef=0.4, splitInto=[1], **kwargs):
        super().__init__(*args, **kwargs)
        self.nsymN = nsymN
        if nsym is None:
            self.nsym = list(range(2, 230))
        else:
            self.nsym = nsym
        self.sym_coef = sym_coef
        self.splitInto = splitInto
        self.fixRndSeed = False

    def generateStructure(self, composition, latVol):
        symbols = list(composition.keys())
        numIons = list(composition.values())
        numIons_tmp = np.copy(numIons)
        failedDist = 0
        nsym = self.nsym[np.random.randint(0, len(self.nsym))]
        self.logger.debug(f"Trying {nsym} symmetry")
        externalPressure = self.config['externalPressure'] if 'externalPressure' in self.config else 0.0001
        volumeType = {'volumeType' : self.config['volumeType']} if 'volumeType' in self.config else {}
        badSymmetryCounter = 0
        startTime = time()
        CenterminDistMatrice = np.zeros((len(symbols), len(symbols)))
        radii = []
        for s in symbols:
            if s not in self.compositionSpace.molecules:
                radii.append(0.22 * calcVolume(externalPressure, s, **volumeType) ** (1.0 / 3.0))
            else:
                molecule = Crystal.fromDICT(self.compositionSpace.molecules[s])
                molecule.set_masses([1] * len(molecule))
                molecule.translate(-molecule.get_center_of_mass())
                values, vectors = molecule.get_moments_of_inertia(vectors=True)
                ind = np.argsort(values)[0]
                short_direction = vectors[ind]
                height_map = [np.abs(np.dot(pos, short_direction)) for pos in molecule.get_positions()]
                ind = np.argsort(height_map)[0]
                s = molecule.get_chemical_symbols()[ind]
                radii.append(
                    0.45 * np.power(calcVolume(externalPressure, s, **volumeType), 1 / 3.0) + height_map[ind])
        for i, j in combinations_with_replacement(range(len(radii)), 2):
            CenterminDistMatrice[i, j] = CenterminDistMatrice[j, i] = (radii[i] + radii[j])

        # minDistMatrice = np.copy(self.config.minDistMatrice)

        while True:
            endTime = time()
            failedTime = endTime - startTime
            # if failedDist > MAX_RANDOM_FAILED_DIST or failedTime > MAX_RANDOM_TIME:
            #     if self.config.minDistMatrice[0][0] > 0.8 * minDistMatrice[0][0]:
            #         if failedTime > MAX_RANDOM_TIME:
            #             self.logger.debug(f'WARNING! Can not generate a structure after {MAX_RANDOM_TIME / 60} minutes. '
            #                                'The minimum distance threshold will be lowered by 10%.')
            #         else:
            #             self.logger.debug(f'WARNING! Can not generate a structure after {MAX_RANDOM_FAILED_DIST} tries. '
            #                                'The minimum distance threshold will be lowered by 10%.')
            #         failedDist = 0
            #         startTime = time()
            #         self.config.minDistMatrice *= 0.9
            #
            #     else:
            #         msg = f'Could not generate a structure after {MAX_RANDOM_FAILED_DIST} tries or {MAX_RANDOM_TIME / 60} minutes.\n'
            #         msg += 'Please check the input files. The calculation has to stop.\n'
            #         msg += 'Possible reasons: unreasonably big IonDistances.\n'
            #         msg += 'Remember they should be much smaller than the real interatomic distances,\n'
            #         msg += 'but not too small for pseudopotential overlap errors to kill interatomic repulsion.\n'
            #         self.logger.debug(msg)
            #         self.config.minDistMatrice = minDistMatrice
            #         raise VOFailed

            if badSymmetryCounter > 15 or (badSymmetryCounter > 5 and sum(self.splitInto) > 3):
                badSymmetryCounter = 0
                # change the symmetry group if can't generate the crystal
                # Pick a random group from those specified by user (different from Matlab implementation):
                nsym = self.nsym[np.random.randint(0, len(self.nsym))]
                self.logger.debug(f"Trying {nsym} symmetry")
            else:
                badSymmetryCounter += 1

            # if sum(self.splitInto) > 3:  # split cell
            #     startLat_in = AtomicStructure(cell=lat1).get_cell_length_and_angles()  # latConverter(lat)
            #     splitInto = self.splitInto[int(np.ceil(len(self.splitInto) * np.random.rand())) - 1]
            #     lat, errorS, candidate = splitBigCell(self.config, self.fixRndSeed, startLat_in, splitInto, numIons, nsym,
            #                                           self.sym_coef)
            # else:

            try:
                candidate, lat, errorS = symope_crystal(CenterminDistMatrice, 'latticeValues' in self.config,
                                                        self.fixRndSeed, nsym, numIons_tmp, latVol, self.sym_coef)
                if errorS == 0:
                    name, cell, coordinates, operations = determineOperations(lat, numIons, candidate)
                    return name, cell, dict(zip(symbols, coordinates)), dict(zip(symbols, operations))
            except Exception as e:
                self.logger.exception(e)

            failedDist += 1
