"""
USPEX.Common.Output
===================

Class handling output

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import logging
from copy import copy

from .Presets import presetOutput
from .NoRepresentation import NoRepresentation


logger = logging.getLogger(__name__)


class Output(object):
    """
    Class handling output.
    """
    def __init__(self, representationFactory=NoRepresentation, **kwargs):
        """
        Initializes the class.

        :param name:
        :param target:
        :param selection:
        :param numParallelCalcs:
        :param stages:
        :param output:
        :param representationFactory:
        :param kwargs:
        """
        self.representation = representationFactory(presetOutput, **kwargs)
        self.optimizer = None

        self.populations = []
        self.infos = []
        self.systems = {}
        self.optimizers = []

    def setOptimizer(self, optimizer):
        self.optimizer = optimizer

    def handleSystem(self, system):
        try:
            ID = system.ID
            system = copy(system)
            system.clean()
            if ID in self.systems:
                self.systems[ID].append(system)
            else:
                self.systems[ID] = [system]
            self.representation.presentSystems(self.systems, self.optimizer)
        except Exception as ex:
            logger.exception(ex)

    def handlePopulation(self, population):
        try:
            self.populations.append(copy(population))
            self.representation.presentOutput(self.populations, self.optimizer)
        except Exception as ex:
            logger.exception(ex)

    def handleInfo(self, info):
        try:
            self.infos.append(copy(info))
            self.representation.presentInfo(self.infos)
        except Exception as ex:
            logger.exception(ex)

    def handleOptimizer(self, optimizer):
        try:
            self.optimizers.append(copy(optimizer))
            self.representation.presentOptimizer(self.optimizers, self.optimizer)
        except Exception as ex:
            logger.exception(ex)
