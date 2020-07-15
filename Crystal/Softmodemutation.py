import logging
logger = logging.getLogger(__name__)

import numpy as np
from collections import namedtuple
from copy import copy

from ..VarOperator import VarOperator, VOFailed
from ..Atomistic.softmodes.calcSoftModes import calcSoftModes


_MIN_VALID_FREQUENCY = 5.0e-4


class Softmodemutation(VarOperator):
    def __init__(self, systemFactory, config, pool, initFrac : float=0.0, minFrac : float=0.1, maxFrac : float=1.0,
                 degree: float = None):
        super().__init__(systemFactory, config, pool, initFrac, minFrac, maxFrac)
        self.degree= degree
        self.knownSystems = {}

    def __call__(self, system):
        if system.ID in self.knownSystems:
            frequencies, eigenVectors = self.knownSystems[system.ID]
        else:
            frequencies, eigenVectors = calcSoftModes(system)
            self.knownSystems[system.ID] = (frequencies, eigenVectors)
        while len(frequencies) > 0:
            freq = frequencies.pop(0)
            eigenVector = eigenVectors.pop(0)
            if freq < _MIN_VALID_FREQUENCY:
                continue
            displacements = eigenVector.reshape((len(system),3))
            degree = self.degree if self.degree else system.covalentRadii.mean() * 3
            displacements *= degree/np.max(np.linalg.norm(displacements, axis = 1))
            newsystem1 = self.systemFactory(cell=system.cell, **self.config)
            newsystem2 = self.systemFactory(cell=system.cell, **self.config)
            for (translation, rotation, _), molecule1 in zip(system.decomposeDisplacements(displacements),
                                                                               system.molecules):

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
            offsprings = ()
            if newsystem1.isGoodSystem() and newsystem1 != system:
                newsystem1.howCome = 'Softmodemutation'
                self.pool.assignID(newsystem1)
                newsystem1.parent = str(system.ID)
                logger.info(f"Structure {newsystem1.ID} created via Softmode mutation at {freq:.4f} mode"
                            f" from {system.ID} parent.")
                offsprings += (newsystem1,)
            if newsystem2.isGoodSystem() and newsystem2 != newsystem1:
                newsystem2.howCome = 'Softmodemutation'
                self.pool.assignID(newsystem2)
                newsystem2.parent = str(system.ID)
                logger.info(f"Structure {newsystem2.ID} created via Softmode mutation at {freq:.4f} mode"
                            f" from {system.ID} parent.")
                offsprings += (newsystem2,)
            if offsprings:
                return offsprings
        raise VOFailed
