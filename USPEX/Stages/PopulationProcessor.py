import logging
import asyncio
import pickle as pcl

from copy import deepcopy, copy
from pathlib import Path
from shutil import copyfile


logger = logging.getLogger(__name__)


class PopulationProcessor:

    def __init__(self, tag, inputKey, **kwargs):
        self.tag = tag
        self.inputKey = inputKey
        self.kwargs = kwargs

    async def run(self, source, sink):
        initial = source[self.inputKey]
        for i, system in enumerate(initial):
            if 'ID' not in system:
                system['ID'] = f'{self.tag}_{i}'
        population, sc = self.initializePopulation(self.tag, initial)
        await self.processPopulation(population=population, **self.kwargs, saveCallback=sc)
        final = [system[-1] for system in population.values()]
        sink.setProperty(self.inputKey, final)

    @staticmethod
    def initializePopulation(tag, initial):
        pd = PopulationDump.load(tag, initial)
        return pd.population, pd.save

    @staticmethod
    async def processPopulation(stages, population, numParallelCalcs, target=None, systems=None, saveCallback=None):
        stages = [Stages.createStage(**stage, target=target) for stage in stages]
        for ID, system in population.items():
            if systems is not None:
                systems[ID] = system[0:1]
        sem = asyncio.Semaphore(numParallelCalcs)
        await asyncio.gather(*(PopulationProcessor.life(system, stages, systems, sem, saveCallback)
                               for system in population.values()))

    @staticmethod
    async def life(processedSystems, stages, systems, sem, saveCallback):
        await sem.acquire()
        ID = processedSystems[0]['ID']
        for i, stage in enumerate(stages):
            assert i < len(processedSystems)
            if i + 1 == len(processedSystems):
                source = processedSystems[-1]
                sink = type(source)(extensions=source.extensions, isBad=False)
                try:
                    await stage.run(source, sink)
                except Exception as ex:
                    logger.warning(f'system {ID} error in relaxation:')
                    logger.exception(ex)
                    sink.setProperty('isBad', True)
                if sink['isBad']:
                    break
                processedSystems.append(deepcopy(sink))
            if systems is not None:
                systems[ID].append(processedSystems[i+1])
            if saveCallback is not None:
                saveCallback()
            # self.populationDump.save()
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
            if ID not in systems:
                systems[ID] = [deepcopy(system)]
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
