import logging
logger = logging.getLogger(__name__)

import numpy as np
import itertools
from copy import copy

from ..VarOperator import VarOperator, VOFailed
from ..Atomistic.calcDefaultVolume import calcVolumeForComposition


def randomPermutation(array, enumerate = False, maxSize = None):
    maxSize = maxSize if maxSize is not None else len(array)
    for i in np.random.permutation(list(range(len(array))))[0:maxSize]:
        yield (i, array[i]) if enumerate else array[i]


class Random(VarOperator):
    '''

    '''

    HOW_MANY_ATTEMPTS_COMPOSITION = 10
    HOW_MANY_ATTEMPTS_ROTATION = 100
    HOW_MANY_POINT_VARIANTS = 10

    logger = logger

    def __init__(self, systemFactory, config, pool, utilities):
        super().__init__(systemFactory, config, pool, utilities)
        self.compositionSpace = utilities['compositionSpace']
        self.structs = None
        self.arxiv = {}

    def tune(self,population):
        pass

    def prepare(self):
        pass

    def standby(self):
        pass


    def generateStructure(self, numIons, latVol):
        pass


    def populateStructure(self, lattice, composition, coordinates, operations):
        symbols = list(composition.keys())
        self.logger.debug("Populationg structure")
        if self.compositionSpace.molecules:
            for i in list(range(self.HOW_MANY_ATTEMPTS_ROTATION)):
                newstructure = self.systemFactory(molecules=[], cell=lattice, **self.config)
                structureIncomplete = False
                try:
                    for symbol in symbols:
                        atomCoordinates = coordinates[symbol]
                        atomOperations = operations[symbol]
                        if symbol in self.compositionSpace.molecules:
                            moleculeRef = self.systemFactory.fromDICT(self.compositionSpace.molecules[symbol])
                            moleculeRef.set_cell(lattice)
                            moleculeRef.rotate((360 * np.random.random_sample()), 'z')
                            # To make sphericaly symmetric distribution we need to get probability of theta angle
                            #  to have some value to be proportional to the radius of respective parallel.
                            theta = np.arcsin(np.sqrt(np.random.random_sample())) * 180 / np.pi
                            if np.random.randint(2):
                                theta = 180 - theta
                            moleculeRef.rotate(theta, 'y')
                            moleculeRef.rotate((360 * np.random.random_sample()), 'z')
                            for nodeCoordinates, groups in zip(atomCoordinates, atomOperations):
                                structureIncomplete = True
                                for group in randomPermutation(groups, maxSize=self.HOW_MANY_POINT_VARIANTS):
                                    if len(group.operators) == len(nodeCoordinates):
                                        nodeOperations = group.operators
                                        for coordinate, operation in zip(nodeCoordinates, nodeOperations):
                                            molecule = copy(moleculeRef)
                                            molecule.set_scaled_positions(np.dot(molecule.get_scaled_positions(), operation[0:3, 0:3]))
                                            molecule.translate_scaled(coordinate)
                                            newstructure.extend(molecule)
                                        structureIncomplete = False
                                        break
                        else:
                            nodeCoordinates = np.vstack(atomCoordinates)
                            newstructure.extend(self.systemFactory(symbols=[symbol] * len(nodeCoordinates),
                                                                   cell=lattice, scaled_positions=nodeCoordinates))
                except Exception as e:
                    self.logger.exception(e)
                    continue

                if structureIncomplete:
                    self.logger.debug("Structure incomplete")
                    break
                elif newstructure.isGoodSystem():
                    return newstructure
            raise VOFailed
        else:
            all_coordinates = np.vstack([*itertools.chain(*coordinates.values())])
            numIons = list(composition.values())
            newstructure = self.systemFactory(symbols=np.repeat(symbols, numIons), cell=lattice,
                                              scaled_positions=all_coordinates, **self.config)
            return newstructure

    def __call__(self) -> tuple:

        for i in list(range(self.HOW_MANY_ATTEMPTS_COMPOSITION)):
            composition = self.compositionSpace.randomComposition()
            self.logger.debug(f'Need {composition} composition')

            latVol = calcVolumeForComposition(composition, **self.config)

            try:
                name, cell, coordinates, operations = self.generateStructure(composition, latVol)
            except VOFailed:
                logger.info(f"Failed generate structure with {composition} composition")
                continue

            if 'latticeValues' in self.config:
                cell = self.config['latticeValues']

            # if not self.config.isGoodCenterDistances(np.repeat(self.config.symbols, numIons),
            #                                          np.vstack([*itertools.chain(*coordinates)]), cell):
            #     self.logger.info("Found structure with {} origin violates molecular center distances".format(name))
            #     continue

            try:
                newstructure = self.populateStructure(cell, composition, coordinates, operations)
            except VOFailed:
                self.logger.info(f"Failed populate structure with {composition} composition and {name} origin")
                continue

            if 'cellVectors' in self.config:
                cellVectors = np.asarray(self.config['cellVectors'], dtype=float)
                if cellVectors.shape == (3,3):
                    newstructure.set_cell(cellVectors)
                else:
                    self.logger.debug(f'Incorrect cellVectors specified in input parameters: {cellVectors}.')
            elif 'cellLengthsAndAngles' in self.config:
                cellLengthsAndAngles = np.asarray(self.config['cellLengthsAndAngles'], dtype=float)
                if cellLengthsAndAngles.shape == (6,):
                    newstructure.set_cell(cellLengthsAndAngles)
                else:
                    self.logger.debug(f'Incorrect cellLengthsAndAngles specified in input parameters: {cellLengthsAndAngles}.')
            elif 'cellVolume' in self.config:
                cellVolume = self.config['cellVolume']
                if isinstance(cellVolume, float):
                    cell = newstructure.cell
                    cell *= cellVolume/np.det(cell)
                    newstructure.set_cell(cell)
                else:
                    self.logger.debug(f'Incorrect cellVolume specified in input parameters: {cellVolume}.')

            if newstructure.isGoodSystem():
                crystal = newstructure
                self.pool.assignID(crystal)
                crystal.howCome = self.__class__.__name__
                crystal.parent = 'None'
                self.logger.info(f"Structure {crystal.ID} created with {composition} composition and {name} origin"
                            f" actual symmetry is {crystal.symmetry}.")
                return (crystal,)

        raise VOFailed
