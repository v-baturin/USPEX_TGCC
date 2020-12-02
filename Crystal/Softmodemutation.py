import logging
logger = logging.getLogger(__name__)

import numpy as np
from collections import namedtuple
from copy import copy

from ..VarOperator import VarOperator, VOFailed
from ..Atomistic.softmodes.calcSoftModes import calcSoftModes


_MIN_VALID_FREQUENCY = 5.0e-4


class Softmodemutation(VarOperator):
    def __init__(self, systemFactory, config, pool, utilities, degree: float = None):
        super().__init__(systemFactory, config, pool, utilities)
        self.degree= degree
        self.knownSystems = {}

    def __call__(self, system):
        if system['ID'] in self.knownSystems:
            frequencies, eigenVectors = self.knownSystems[system['ID']]
        else:
            frequencies, eigenVectors = calcSoftModes(system['structure'])
            self.knownSystems[system['ID']] = (frequencies, eigenVectors)
        while len(frequencies) > 0:
            freq = frequencies.pop(0)
            eigenVector = eigenVectors.pop(0)
            if freq < _MIN_VALID_FREQUENCY:
                continue
            displacements = eigenVector.reshape((len(system['structure']),3))
            degree = self.degree if self.degree else system['structure'].covalentRadii.mean() * 3
            displacements *= degree/np.max(np.linalg.norm(displacements, axis = 1))
            newsystem1 = self.systemFactory(cell=system['structure'].cell, **self.config)
            newsystem2 = self.systemFactory(cell=system['structure'].cell, **self.config)
            for (translation, rotation, _), molecule1 in zip(system['structure'].decomposeDisplacements(displacements),
                                                                               system['structure'].molecules):

                molecule2 = copy(molecule1)
                angle = np.linalg.norm(rotation)
                if  angle > 0.001:
                    axis = rotation / angle
                    molecule1.rotate( angle, axis, center='COP')
                    molecule2.rotate(-angle, axis, center='COP')
                molecule1.translate( translation)
                molecule2.translate(-translation)
                newsystem1.extend(molecule1)
                newsystem2.extend(molecule2)

                if 'cellVectors' in self.config:
                    cellVectors = np.asarray(self.config['cellVectors'], dtype=float)
                    if cellVectors.shape == (3, 3):
                        newsystem1.set_cell(cellVectors, scale_atoms=True)
                        newsystem2.set_cell(cellVectors, scale_atoms=True)
                    else:
                        logger.debug(f'Incorrect cellVectors specified in input parameters: {cellVectors}.')
                elif 'cellLengthsAndAngles' in self.config:
                    cellLengthsAndAngles = np.asarray(self.config['cellLengthsAndAngles'], dtype=float)
                    if cellLengthsAndAngles.shape == (6,):
                        newsystem1.set_cell(cellLengthsAndAngles, scale_atoms=True)
                        newsystem2.set_cell(cellLengthsAndAngles, scale_atoms=True)
                    else:
                        logger.debug(
                            f'Incorrect cellLengthsAndAngles specified in input parameters: {cellLengthsAndAngles}.')
                elif 'cellVolume' in self.config:
                    cellVolume = self.config['cellVolume']
                    if isinstance(cellVolume, float):
                        cell = newsystem1.cell
                        cell *= (cellVolume/np.linalg.det(cell))**(1.0/3.0)
                        newsystem1.set_cell(cell, scale_atoms=True)
                        cell = newsystem2.cell
                        cell *= (cellVolume/np.linalg.det(cell))**(1.0/3.0)
                        newsystem2.set_cell(cell, scale_atoms=True)
                    else:
                        logger.debug(f'Incorrect cellVolume specified in input parameters: {cellVolume}.')

            offsprings = ()
            newsystem1 = {'structure': newsystem1}
            newsystem2 = {'structure': newsystem2}
            if newsystem1['structure'].isGoodSystem() and newsystem1['structure'] != system['structure']:
                newsystem1['howCome'] = self.__class__.__name__
                self.pool.assignID(newsystem1)
                newsystem1['parent'] = str(system['ID'])
                logger.info(f"Structure {newsystem1['ID']} created via Softmode mutation at {freq:.4f} mode"
                            f" from {system['ID']} parent.")
                offsprings += (newsystem1,)
            if newsystem2['structure'].isGoodSystem() and newsystem2['structure'] != newsystem1['structure']:
                newsystem2['howCome'] = self.__class__.__name__
                self.pool.assignID(newsystem2)
                newsystem2['parent'] = str(system['ID'])
                logger.info(f"Structure {newsystem2['ID']} created via Softmode mutation at {freq:.4f} mode"
                            f" from {system['ID']} parent.")
                offsprings += (newsystem2,)
            if offsprings:
                return offsprings
        raise VOFailed
