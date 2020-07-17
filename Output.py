"""
USPEX.Common.Output
===================

Class handling output

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import logging
from copy import copy

from .Worker import Worker
from .NoRepresentation import NoRepresentation


logger = logging.getLogger(__name__)


class Output(object):
    """
    Class handling output.
    """
    def __init__(self, output: dict = None, representationFactory=NoRepresentation, **kwargs):
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
        if output is None:
            output = {}
        self.representation = representationFactory(**kwargs, **output)

        self.populations = []
        self.infos = []
        self.systems = {}
        self.optimizers = []

    def handleSystem(self, system):
        try:
            ID = system.ID
            if ID in self.systems:
                self.systems[ID].append(copy(system))
            else:
                self.systems[ID] = [copy(system)]
            optimizer = self.optimizers[-1] if self.optimizers else None
            self.representation.presentSystems(self.systems, optimizer)
        except Exception as ex:
            logger.exception(ex)

    def handlePopulation(self, population):
        try:
            self.populations.append(copy(population))
            optimizer = self.optimizers[-1] if self.optimizers else None
            self.representation.presentOutput(self.populations, optimizer)
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
            self.representation.presentOptimizer(self.optimizers)
        except Exception as ex:
            logger.exception(ex)
