import logging
logger = logging.getLogger(__name__)

import numpy as np
import itertools
from copy import copy

from ..VarOperator import VarOperator, VOFailed


class Rotation(VarOperator):
    '''

    '''


    HOW_MANY_ATTEMPTS_ROTATION = 100


    def tune(self,population):
        pass

    def prepare(self):
        pass

    def standby(self):
        pass

    def __call__(self, parent, moleculesToRotate = None, axesToRotate = None,
                 dihedralToRotate = None, principleAngles = None, dihedralAngles = None) -> tuple:
        randomize = moleculesToRotate is None
        for i in range(self.HOW_MANY_ATTEMPTS_ROTATION):
            newstructure = copy(parent)
            if randomize:
                totalNumMols = len(parent.molecules)
                moleculesToRotate = np.random.choice(totalNumMols, np.random.randint(totalNumMols), replace=False)
                axesToRotate = np.random.randint(3, size = len(moleculesToRotate))
                # TODO add proper dihedrals
                dihedralToRotate = [None]*len(moleculesToRotate)
                principleAngles = np.random.randint(-45, 45, size = len(moleculesToRotate))
                dihedralAngles = np.random.randint(-np.pi/2, np.pi/2, size = len(moleculesToRotate))
            for molecule, axis, dihedral, principleAngle, dihedralAngle in \
                    zip(moleculesToRotate, axesToRotate, dihedralToRotate, principleAngles, dihedralAngles):
                newstructure.rotatePrinciple(molecule, axis, principleAngle)
                if dihedral is not None:
                    try:
                        newstructure.rotateFlexDiherdal(molecule, dihedral, dihedralAngle)
                    except RuntimeError as e:
                        logger.exception(e)

                if 'cellVectors' in self.config:
                    cellVectors = np.asarray(self.config['cellVectors'], dtype=float)
                    if cellVectors.shape == (3, 3):
                        newstructure.set_cell(cellVectors, scale_atoms=True)
                    else:
                        logger.debug(f'Incorrect cellVectors specified in input parameters: {cellVectors}.')
                elif 'cellLengthsAndAngles' in self.config:
                    cellLengthsAndAngles = np.asarray(self.config['cellLengthsAndAngles'], dtype=float)
                    if cellLengthsAndAngles.shape == (6,):
                        newstructure.set_cell(cellLengthsAndAngles, scale_atoms=True)
                    else:
                        logger.debug(
                            f'Incorrect cellLengthsAndAngles specified in input parameters: {cellLengthsAndAngles}.')
                elif 'cellVolume' in self.config:
                    cellVolume = self.config['cellVolume']
                    if isinstance(cellVolume, float):
                        cell = newstructure.cell
                        cell *= (cellVolume/np.linalg.det(cell))**(1/3)
                        newstructure.set_cell(cell, scale_atoms=True)
                    else:
                        logger.debug(f'Incorrect cellVolume specified in input parameters: {cellVolume}.')

            if newstructure.isGoodSystem():
                crystal = newstructure
                self.pool.assignID(crystal)
                crystal.howCome = self.__class__.__name__
                crystal.parent = 'None'
                logger.info(f"Structure {crystal.ID} created via rotation from {parent.ID} parent")
                return (crystal,)

        raise VOFailed
