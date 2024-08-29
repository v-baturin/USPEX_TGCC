import logging
import asyncio


logger = logging.getLogger(__name__)


class PopulationProcessor:

    def __init__(self, tag, source, population, stages, target, **kwargs):
        self.tag = tag
        self.population = population
        self.stages = stages
        self.target = target
        self.source = source
        self.kwargs = kwargs

    async def run(self, system):
        population = self.target.creatPool()
        for system in system.getProperty(self.population, suffix=self.source):
            population.newEntry(system)
        await self.processPopulation(population=population, target=self.target, stages=self.stages, **self.kwargs)
        final = [population.getEntry(ID).getFlavour(self.stages[-1].tag) for ID in population.getIDs()]
        system.setProperty(self.population, final, suffix=self.tag)

    @staticmethod
    async def processPopulation(stages, population, numParallelCalcs, target=None, shellEnv=None):
        stages = [Stages.createStage(**stage, target=target) for stage in stages]
        sem = asyncio.Semaphore(numParallelCalcs)
        await asyncio.gather(*(PopulationProcessor.life(population.getEntry(ID), stages, sem, shellEnv)
                               for ID in population.getIDs()))

    @staticmethod
    async def life(system, stages, sem, shellEnv=None):
        await sem.acquire()
        for stage in stages:
            if stage.tag not in system.flavours and stage.source in system.flavours \
                    and not system.getProperty('isBad', suffix=stage.source):
                await stage.run(system, shellEnv)
        sem.release()

class Stages:

    knownStages = {}

    @classmethod
    def registerStage(cls, name, stageType: type):
        assert name not in cls.knownStages
        cls.knownStages[name] = stageType

    @classmethod
    def createStage(cls, stageType, **kwargs):
        return cls.knownStages[stageType](**kwargs)
