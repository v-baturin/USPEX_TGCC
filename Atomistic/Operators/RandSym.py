import logging
logger = logging.getLogger(__name__)


import numpy as np
import spglib
from copy import copy
from itertools import combinations_with_replacement

from .symope.splitBigCell import splitBigCell
from .symope.symope_crystal import symope_crystal
from time import time

from ..Transformation import Transformation
from ...SpaceGroups.SpaceGroups3D import Group

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



class RandSym:
    def __init__(self, utilities, nsymN=False, nsym=None, sym_coef=0.4, splitInto=[1]):
        self.cellUtility = utilities.cellUtility
        self.compositionSpace = utilities.compositionSpace
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.ionDistances = utilities.ionDistances
        self.conditions = utilities.conditions
        self.nsymN = nsymN
        if nsym is None:
            self.nsym = list(range(2, 230))
        else:
            self.nsym = nsym
        self.sym_coef = sym_coef
        self.splitInto = splitInto
        self.fixRndSeed = False

    def __call__(self, *args, **kwargs):
        composition = self.compositionSpace.randomComposition()

        symbols = list(composition.keys())
        numIons = list(composition.values())
        numIons_tmp = np.copy(numIons)
        failedDist = 0
        nsym = self.nsym[np.random.randint(0, len(self.nsym))]
        logger.debug(f"Trying {nsym} symmetry")
        badSymmetryCounter = 0
        startTime = time()
        CenterminDistMatrice = np.zeros((len(symbols), len(symbols)))
        radii = []
        for s in symbols:
            molecule = self.simpleMoleculeUtility.molecules[s]
            if len(molecule) == 1:
                atomRaduis = self.conditions.calcAtomVolume(s) ** (1.0 / 3.0)
                radii.append(0.22 * atomRaduis)
            else:
                molecule = Transformation.fromRotVector([0.,0.,0.],
                                                        -molecule.getCenterOfMassCartesianCoordinates()).transform(molecule)
                values, vectors = molecule.getPrincipleAxes()
                short_direction = vectors[np.argmin(values)]
                height_map = [np.abs(np.dot(pos, short_direction)) for pos in molecule.getCartesianCoordinates()]
                ind = np.argmin(height_map)
                atomRaduis = self.conditions.calcAtomVolume(molecule.getAtomTypes())[ind] ** (1.0 / 3.0)
                radii.append(0.45 * atomRaduis + height_map[ind])
        for i, j in combinations_with_replacement(range(len(radii)), 2):
            CenterminDistMatrice[i, j] = CenterminDistMatrice[j, i] = (radii[i] + radii[j])
        originalCenterminDistMatrice = copy(CenterminDistMatrice)

        while True:
            endTime = time()
            failedTime = endTime - startTime
            if failedDist > MAX_RANDOM_FAILED_DIST or failedTime > MAX_RANDOM_TIME:
                if CenterminDistMatrice[0][0] > 0.8 * originalCenterminDistMatrice[0][0]:
                    if failedTime > MAX_RANDOM_TIME:
                        logger.debug(f'WARNING! Can not generate a structure after {MAX_RANDOM_TIME / 60} minutes. '
                                           'The minimum distance threshold will be lowered by 10%.')
                    else:
                        logger.debug(f'WARNING! Can not generate a structure after {MAX_RANDOM_FAILED_DIST} tries. '
                                           'The minimum distance threshold will be lowered by 10%.')
                    failedDist = 0
                    startTime = time()
                    CenterminDistMatrice *= 0.9

                else:
                    msg = f'Could not generate a structure after {MAX_RANDOM_FAILED_DIST} tries or {MAX_RANDOM_TIME / 60} minutes.\n'
                    msg += 'Please check the input files. The calculation has to stop.\n'
                    msg += 'Possible reasons: unreasonably big IonDistances.\n'
                    msg += 'Remember they should be much smaller than the real interatomic distances,\n'
                    msg += 'but not too small for pseudopotential overlap errors to kill interatomic repulsion.\n'
                    logger.debug(msg)
                    raise RuntimeError("RandSym failed.")

            if badSymmetryCounter > 15 or (badSymmetryCounter > 5 and sum(self.splitInto) > 3):
                badSymmetryCounter = 0
                # change the symmetry group if can't generate the crystal
                # Pick a random group from those specified by user (different from Matlab implementation):
                nsym = self.nsym[np.random.randint(0, len(self.nsym))]
                logger.debug(f"Trying {nsym} symmetry")
            else:
                badSymmetryCounter += 1


            try:
                if sum(self.splitInto) > 3:  # split cell
                    lat = self.cellUtility.getRandomCell(composition, self.conditions).getCellParameters()
                    lat, candidate = splitBigCell(CenterminDistMatrice, False, self.fixRndSeed, lat,
                                                  np.random.choice(self.splitInto, 1), numIons, nsym, self.sym_coef)
                else:

                    candidate, lat = symope_crystal(CenterminDistMatrice, False, self.fixRndSeed, nsym, numIons_tmp,
                                                    self.cellUtility.getCellVolume(composition, self.conditions),
                                                    self.sym_coef)
                name, cell, coordinates, operations = determineOperations(lat, numIons, candidate)
                cell = self.cellUtility.adjustCell(cell, composition, self.conditions)
                molecules = self.simpleMoleculeUtility.populateStructure(cell, coordinates, operations)
                atomSymbols, atomDistances = self.simpleMoleculeUtility.getMinDistances(molecules, cell)
                minDistMatrix = self.ionDistances.getDistances(atomSymbols, self.conditions)
                if atomDistances >= minDistMatrix:
                    return ({'molecules': molecules, 'cell': cell},)
            except Exception as e:
                logger.exception(e)

            failedDist += 1
