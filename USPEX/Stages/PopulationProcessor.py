import logging
import asyncio


logger = logging.getLogger(__name__)


class PopulationProcessor:

    def __init__(self, tag, source, inputKey, **kwargs):
        self.tag = tag
        self.inputKey = inputKey
        self.source = source
        self.kwargs = kwargs

    async def run(self, system):
        population = system.getProperty(self.inputKey, suffix=self.source)
        for i, system in enumerate(population):
            if 'ID' not in system:
                system['ID'] = f'{self.tag}_{i}'
        await self.processPopulation(population=population, **self.kwargs)
        final = [system[-1] for system in population]
        system.setProperty(self.inputKey, final, suffix=self.tag)

    @staticmethod
    async def processPopulation(stages, population, numParallelCalcs, target=None):
        stages = [Stages.createStage(**stage, target=target) for stage in stages]
        sem = asyncio.Semaphore(numParallelCalcs)
        await asyncio.gather(*(PopulationProcessor.life(system, stages, sem)
                               for system in population))

    @staticmethod
    async def life(system, stages, sem):
        await sem.acquire()
        for stage in stages:
            if stage.tag not in system.flavours:
                # system.setProperty('isBad', False, suffix=stage.tag)
                try:
                    await stage.run(system)
                except Exception as ex:
                    logger.warning(f'system {system.ID} error in relaxation:')
                    logger.exception(ex)
                    system.setProperty('isBad', True, suffix=stage.tag)
                # if system.getProperty('isBad', suffix=stage.tag):
                #     break
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
