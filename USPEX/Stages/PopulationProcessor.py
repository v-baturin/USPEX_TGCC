import logging
import asyncio
import pickle as pcl

from copy import deepcopy, copy
from pathlib import Path
from shutil import copyfile


logger = logging.getLogger(__name__)


class PopulationProcessor:

    def __init__(self, tag, source, inputKey, **kwargs):
        self.tag = tag
        self.inputKey = inputKey
        self.source = source
        self.kwargs = kwargs

    async def run(self, system):
        initial = system.getProperty(self.inputKey, suffix=self.source)
        for i, system in enumerate(initial):
            if 'ID' not in system:
                system['ID'] = f'{self.tag}_{i}'
        population, sc = self.initializePopulation(self.tag, initial)
        await self.processPopulation(population=population, **self.kwargs, saveCallback=sc)
        final = [system[-1] for system in population.values()]
        system.setProperty(self.inputKey, final, suffix=self.tag)

    @staticmethod
    def initializePopulation(tag, initial):
        pd = PopulationDump.load(tag, initial)
        return pd.population, pd.save

    @staticmethod
    async def processPopulation(stages, population, numParallelCalcs, target=None, saveCallback=None):
        stages = [Stages.createStage(**stage, target=target) for stage in stages]
        sem = asyncio.Semaphore(numParallelCalcs)
        await asyncio.gather(*(PopulationProcessor.life(system, stages, sem, saveCallback)
                               for system in population.values()))

    @staticmethod
    async def life(system, stages, sem, saveCallback):
        await sem.acquire()
        ID = system['ID']
        for stage in stages:
            if stage.tag not in system.system:
                system.setProperty('isBad', False, suffix=stage.tag)
                try:
                    await stage.run(system)
                except Exception as ex:
                    logger.warning(f'system {ID} error in relaxation:')
                    logger.exception(ex)
                    system.setProperty('isBad', True, suffix=stage.tag)
                # if system.getProperty('isBad', suffix=stage.tag):
                #     break
            if saveCallback is not None:
                saveCallback()
        sem.release()


class PopulationDump:

    DUMPFILE_TEMPLATE = '{}.dump'
    BACKUP_TEMPLATE = '{}.back'

    def __init__(self, population, dumpFilename):
        self.population = population
        self.dumpFilename = Path(dumpFilename)
        self.dumpFilenameBackup = Path(self.BACKUP_TEMPLATE.format(dumpFilename))

    @staticmethod
    def load(tag, population):
        dumpFilename = Path(PopulationDump.DUMPFILE_TEMPLATE.format(tag))
        if dumpFilename.exists():
            with open(dumpFilename, 'rb') as f:
                systems = pcl.load(f)
            if not set(systems.keys()) <= set(system['ID'] for system in population):
                dumpFilename.unlink()
                systems = {}
        else:
            systems = {}
        for system in population:
            ID = system['ID']
            if ID in systems:
                system.update(systems[ID])
            systems[ID] = system
        return PopulationDump(systems, dumpFilename)

    def save(self):
        if self.dumpFilename.exists():
            copyfile(self.dumpFilename, self.dumpFilenameBackup)
        with open(self.dumpFilename, 'wb') as f:
            pcl.dump(self.population, f)

    def __getitem__(self, item):
        return self.population[item]

    def __setitem__(self, key, value):
        self.population[key] = value

    def __contains__(self, item):
        return item in self.population

class Stages:

    knownStages = {}

    @classmethod
    def registerStage(cls, name, stageType: type):
        assert name not in cls.knownStages
        cls.knownStages[name] = stageType

    @classmethod
    def createStage(cls, stageType, **kwargs):
        return cls.knownStages[stageType](**kwargs)
