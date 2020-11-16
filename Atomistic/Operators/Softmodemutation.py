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
