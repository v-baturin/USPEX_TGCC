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
ATTEMPTS_ROTATION = 1


def determineOperations(lat, numIons, candidate):
    # logger.debug("Determinig operations variants")
    # cell = (lat, candidate, np.repeat(list(range(len(numIons))), numIons))
    # dataset = spglib.get_symmetry_dataset(cell, symprec=1e-2)
    # symbol = dataset['international']
    # group = Group.getGroupFromSymbol(symbol)
    # equivalent_atoms = dataset['equivalent_atoms']
    # coordinates = []
    # operations = []
    # i = 0
    # for n in numIons:
    #     atomCoordinates = []
    #     atomOperations = []
    #     equivalent_atoms_one_type = equivalent_atoms[i:i + n]
    #     for index in np.unique(equivalent_atoms_one_type):
    #         nodeIndices = np.where(equivalent_atoms==index)[0]
    #         assert len(nodeIndices) == len(np.where(equivalent_atoms_one_type==index)[0])
    #         atomCoordinates.append(candidate[nodeIndices])
    #         atomOperations.append(group.getNotPositionInvariantSubgroups(candidate[nodeIndices[0]]))
    #     coordinates.append(atomCoordinates)
    #     operations.append(atomOperations)
    #     i += n

    operations = []
    operation = np.eye(4, dtype=float)
    offset = 0
    for n in numIons:
        tmp_operations = []
        for i in range(n):
            operation[0:3, 3] = candidate[i + offset]
            tmp_operations.append(np.copy(operation))
        operations.append([[tmp_operations]])
        offset += n

    return None, lat, operations



class RandSym:
    def __init__(self, utilities, nsymN=False, nsym=None, sym_coef=0.4, splitInto=[1],
                 attemptsRotation: int = ATTEMPTS_ROTATION, debug = False):
        self.cellUtility = utilities.cellUtility
        self.environmentUtility = utilities.environmentUtility
        self.compositionSpace = utilities.compositionSpace
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.ionDistances = utilities.ionDistances
        self.bonds = utilities.bonds
        self.conditions = utilities.conditions
        self.nsymN = nsymN
        if nsym is None:
            self.nsym = list(range(2, 231))
        elif isinstance(nsym, str):
            self.nsym = list(parseIntSet(nsym))
        elif isinstance(nsym, list):
            self.nsym = nsym
        else:
            RuntimeError(f'Wrong type of nsym parameter: {type(nsym)}')
        self.sym_coef = sym_coef
        self.splitInto = splitInto
        self.attemptsRotation = attemptsRotation
        if debug:
            logger.setLevel(logging.DEBUG)

        self.fixRndSeed = False

    def __call__(self, *args, **kwargs):
        composition = self.compositionSpace.randomComposition()

        symbols = list(composition.keys())
        numIons = list(composition.values())
        numIons_tmp = np.copy(numIons)
        failedDist = 0
        nsym, = np.random.choice(self.nsym, 1)
        logger.debug(f"Trying {nsym} symmetry")
        badSymmetryCounter = 0
        startTime = time()
        centerMinDistMatrix = np.zeros((len(symbols), len(symbols)))
        radii = []
        for s in symbols:
            molecule = self.simpleMoleculeUtility.molecules[s]
            if len(molecule) == 1:
                atomRaduis = self.ionDistances.volumeEstimator.calcAtomVolume(s, self.conditions.externalPressure) ** (1.0 / 3.0)
                radii.append(0.22 * atomRaduis)
            else:
                molecule = Transformation.fromRotVector([0.,0.,0.],
                                                        -molecule.getCenterOfMassCartesianCoordinates()).transform(molecule)
                values, vectors = molecule.getPrincipalAxes()
                short_direction = vectors[np.argmin(values)]
                height_map = [np.abs(np.dot(pos, short_direction)) for pos in molecule.getCartesianCoordinates()]
                ind = np.argmin(height_map)
                atomRaduis = self.ionDistances.volumeEstimator.calcAtomVolume(molecule.getAtomTypes()[ind], self.conditions.externalPressure) ** (1.0 / 3.0)
                radii.append(0.45 * atomRaduis + height_map[ind])
        for i, j in combinations_with_replacement(range(len(radii)), 2):
            centerMinDistMatrix[i, j] = centerMinDistMatrix[j, i] = (radii[i] + radii[j])
        distCoeff = 1.0

        while True:
            envAssembler = np.random.choice(self.environmentUtility.assemblers) if self.environmentUtility.assemblers\
                else None
            envCell = envAssembler.getCell() if envAssembler is not None else None

            endTime = time()
            failedTime = endTime - startTime
            if failedDist > MAX_RANDOM_FAILED_DIST or failedTime > MAX_RANDOM_TIME:
                if distCoeff > 0.8:
                    if failedTime > MAX_RANDOM_TIME:
                        logger.debug(f'WARNING! Can not generate a structure after {MAX_RANDOM_TIME / 60} minutes. '
                                           'The minimum distance threshold will be lowered by 10%.')
                    else:
                        logger.debug(f'WARNING! Can not generate a structure after {MAX_RANDOM_FAILED_DIST} tries. '
                                           'The minimum distance threshold will be lowered by 10%.')
                    failedDist = 0
                    startTime = time()
                    distCoeff *= 0.9

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
                nsym, = np.random.choice(self.nsym, 1)
                logger.debug(f"Trying {nsym} symmetry")
            else:
                badSymmetryCounter += 1


            try:
                estimatedVolume = self.cellUtility.getCellVolume()
                if estimatedVolume is None:
                    elementalComposition = self.simpleMoleculeUtility.getElementalComposition(composition)
                    estimatedVolume = self.ionDistances.volumeEstimator.calcCompositionVolume(elementalComposition,
                                                                                              self.conditions.externalPressure)
                if sum(self.splitInto) > 3:  # split cell
                    lat = self.cellUtility.getRandomCell(estimatedVolume, sum(numIons),
                                                         baseCell=envCell).getCellParameters()
                    lat, candidate = splitBigCell(distCoeff * centerMinDistMatrix, False, self.fixRndSeed, lat,
                                                  np.random.choice(self.splitInto), numIons, nsym, self.sym_coef)
                else:

                    candidate, lat = symope_crystal(distCoeff * centerMinDistMatrix, False, self.fixRndSeed, nsym, numIons_tmp,
                                                    estimatedVolume, self.sym_coef)
                name, cell, operations = determineOperations(lat, numIons, candidate)
                operations = dict(zip(symbols, operations))
                cell = self.cellUtility.adjustCell(cell, estimatedVolume, sum(numIons), baseCell=envCell)
                for i in range(self.attemptsRotation):
                    offspring = self.simpleMoleculeUtility.populateStructure(cell, operations)
                    if envAssembler is not None:
                        offspring['environment'] = envAssembler.assemble(**offspring)
                    atomSymbols, atomDistances, disassembler = self.simpleMoleculeUtility.getMinDistances(**offspring)
                    minDistMatrix = self.ionDistances.getDistances(atomSymbols, self.conditions.externalPressure)
                    if disassembler.environment is not None:
                        inds = disassembler.envIndices
                        atomDistances[tuple(np.meshgrid(inds, inds))] = minDistMatrix[tuple(np.meshgrid(inds, inds))]
                    if np.all(atomDistances >= distCoeff * minDistMatrix):
                        self.conditions.putConditions(offspring)
                        structure, disassembler = self.simpleMoleculeUtility.atomicDisassemblerType.assemble(**offspring)
                        if self.bonds.isConnected(structure):
                            return offspring,
            except Exception as e:
                logger.debug(e, exc_info=True)

            failedDist += 1

def parseIntSet(nputstr=""):
  selection = set()
  invalid = set()
  # tokens are comma seperated values
  tokens = [x.strip() for x in nputstr.split(',')]
  for i in tokens:
     try:
        # typically tokens are plain old integers
        selection.add(int(i))
     except:
        # if not, then it might be a range
        try:
           token = [int(k.strip()) for k in i.split('-')]
           if len(token) > 1:
              token.sort()
              # we have items seperated by a dash
              # try to build a valid range
              first = token[0]
              last = token[len(token)-1]
              for x in range(first, last+1):
                 selection.add(x)
        except:
           # not an int and not a range...
           invalid.add(i)
  return selection