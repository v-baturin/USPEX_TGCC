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


class Output(Worker):
    """
    Class handling output.
    """
    def __init__(self, name: str, target: dict, selection: dict, numParallelCalcs: int, stages: list,
                 output: dict = None, representationFactory=NoRepresentation, **kwargs):
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
        self.numParallelCalcs = numParallelCalcs
        self.numStages = len(stages)

        if output is None:
            output = {}
        self.representation = representationFactory(name, selection['type'], target['type'], **output)

        if 'fitness' in target:
            self.fitness = target['fitness']
        else:
            self.fitness = None
        self.populations = []
        self.analyses = []
        self.systems = {}
        self.pools = []

        self.selectionConfig = None
        self.targetConfig = None

    def handleSystem(self, system):
        try:
            ID = system.ID
            if ID in self.systems:
                self.systems[ID].append(copy(system))
            else:
                self.systems[ID] = [copy(system)]
            self.representation.presentSystems(self.systems, self.numStages, self.fitness)
        except Exception as ex:
            logger.exception(ex)

    def handlePopulation(self, population):
        try:
            self.populations.append(copy(population))
            self.representation.presentOutput(self.targetConfig, self.selectionConfig, self.numStages,
                                              self.numParallelCalcs, self.populations, self.fitness)
        except Exception as ex:
            logger.exception(ex)

    def handleAnalysis(self, analysis):
        try:
            self.analyses.append(copy(analysis))
            self.representation.presentAnalysis(self.analyses)
        except Exception as ex:
            logger.exception(ex)

    def handlePool(self, pool):
        try:
            self.pools.append(copy(pool))
            self.representation.presentPool(self.pools, self.fitness)
        except Exception as ex:
            logger.exception(ex)
