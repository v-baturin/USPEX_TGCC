import logging
logger = logging.getLogger(__name__)

import numpy as np

from itertools import combinations_with_replacement
from ..Transformation import Transformation

from time import time

from ....Common.Atomistic.CellUtility import Cell

from pyxtal import pyxtal

MAX_RANDOM_FAILED_DIST = 10000
MAX_RANDOM_TIME = 300
ATTEMPTS_ROTATION = 1

class RandSym0D:
    def __init__(self, utilities, nsymN=False, nsym=None,
                 attemptsRotation: int = ATTEMPTS_ROTATION):
        self.cellUtility = utilities.cellUtility
        self.compositionSpace = utilities.compositionSpace
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.ionDistances = utilities.ionDistances
        self.conditions = utilities.conditions
        if self.simpleMoleculeUtility.isTrueMolecular:
            raise RuntimeError("RandSym0D does not currently work in molecular regime.")
        self.nsymN = nsymN
        if nsym is None:
            self.nsym = list(range(1, 59))
        else:
            self.nsym = nsym
        self.attemptsRotation = attemptsRotation

    def __call__(self, *args, **kwargs):
        composition = self.compositionSpace.randomComposition()

        symbols = list(composition.keys())
        numIons = list(composition.values())
        nsym, = np.random.choice(self.nsym, 1)
        logger.debug(f"Trying {nsym} symmetry")
        badSymmetryCounter = 0
        startTime = time()

        structurePyxtal = pyxtal()
        try:
            structurePyxtal.from_random(0, nsym, symbols, numIons, max_count = 1)
        except Exception as e:
            logger.debug(e, exc_info=True)

        ase_struc = structurePyxtal.to_ase()
        cell = structurePyxtal.lattice.matrix * 2.1
        ase_struc.set_cell(cell)
        ase_struc.center(about=(cell[0,0] / 2.0, cell[1,1] / 2.0, cell[2,2] / 2.0))

        elementalComposition = self.simpleMoleculeUtility.getElementalComposition(composition)

        cell = Cell(cell, (0,0,0))

        # Convert the ase coordinates to USPEX format
        candidate = ase_struc.get_scaled_positions()
        coordinates = []
        offset = 0
        for n in numIons:
            tmp_coordinates = []
            for i in range(n):
                tmp_coordinates.append(np.array([candidate[i + offset]]))
            coordinates.append(tmp_coordinates)
            offset += n

        operations = [None] * len(coordinates) # The result does not look to have the right shape
        coordinates = dict(zip(symbols, coordinates))
        operations = dict(zip(symbols, operations))
        molecules = self.simpleMoleculeUtility.populateStructure(cell, coordinates, operations)
        system = {'molecules': molecules, 'cell': cell}
        self.conditions.putConditions(system)
        return (system,)