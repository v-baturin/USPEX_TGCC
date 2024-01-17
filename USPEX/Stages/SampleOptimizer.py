"""
USPEX.Common.SampleOptimizer
============================

Class implementing external optimizer

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import logging
from itertools import chain

from ..Optimizers.PoolEntry import FlavourFactory, PoolEntry


logger = logging.getLogger(__name__)


class SampleOptimizer(object):

    Stages = None

    @classmethod
    def setStages(cls, Stages):
        cls.Stages = Stages

    def __init__(self, sample, source, stages, **kwargs):
        self.sample = sample
        self.source = source
        self.stages = stages
        self.flavourFactory = FlavourFactory({})

    async def update(self, population):
        IDs = population.getIDs()
        goodPopulation = []
        for ID in IDs:
            system = population.getEntry(ID)
            if not system.getProperty('isBad', suffix=self.source):
                goodPopulation.append(system)
        sample = list(chain(*(individual.getProperty(self.sample, suffix=self.source) for individual in goodPopulation)))
        if len(sample) == 0:
            return None, True, False
        else:
            stages = [self.Stages.createStage(**stage) for stage in self.stages]
            system = PoolEntry.newEntry(self.flavourFactory(sample=sample, isBad=False))
            for stage in stages:
                if stage.tag not in system.flavours and stage.source in system.flavours \
                        and not system.getProperty('isBad', suffix=stage.source):
                    await stage.run(system)
            return None, False, False
