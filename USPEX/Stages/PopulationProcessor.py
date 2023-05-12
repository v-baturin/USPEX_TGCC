import logging
import asyncio
import pickle as pcl

from copy import deepcopy, copy
from pathlib import Path
from shutil import copyfile


logger = logging.getLogger(__name__)


class PopulationProcessor:

    stages = None

    @classmethod
    def setStages(cls, stagesType):
        cls.stages = stagesType

    def __init__(self, tag, stages, inputKey, numParallelCalcs, systems=None, checkCallback=None, **kwargs):
        self.tag = tag
        self.stages = [self.stages.createStage(**stage) for stage in stages]
        self.inputKey = inputKey
        self.numParallelCalcs = numParallelCalcs
        self.systems = systems
        self.checkCallback = checkCallback

    async def run(self, system):
        population = system[self.inputKey]
        for i, subsystem in enumerate(population):
            if 'ID' not in subsystem:
                subsystem['ID'] = f'{system["ID"]}_{i}'
        populationDump = PopulationDump.load(system['ID'], self.tag, population)
        sem = asyncio.Semaphore(self.numParallelCalcs)
        population = await asyncio.gather(*(self.life(system, populationDump, sem) for system in population))
        system = copy(system)
        system[self.inputKey] = population
        return system

    async def life(self, system, populationDump, sem):
        await sem.acquire()
        ID = system['ID']
        system['isBad'] = False
        if ID not in populationDump:
            populationDump[ID] = [deepcopy(system)]
        processedSystems = populationDump[ID]
        if self.systems is not None:
            self.systems[ID] = processedSystems[0:1]
        for i, stage in enumerate(self.stages):
            if i + 1 < len(processedSystems):
                system.update(processedSystems[i + 1])
            else:
                try:
                    system = await stage.run(system)
                except Exception as ex:
                    logger.warning(f'system {ID} error in relaxation:')
                    logger.exception(ex)
                    system['isBad'] = True
                    break
                if self.checkCallback is not None and not self.checkCallback(system):
                    logger.info(f'system {ID} violates constraints')
                    system['isBad'] = True
                    break
                processedSystems.append(deepcopy(system))
            if self.systems is not None:
                self.systems[ID].append(processedSystems[i+1])
            populationDump.save()
        sem.release()
        return system


class PopulationDump:

    DUMPFILE_TEMPLATE = '{}_{}.dump'
    BACKUP_TEMPLATE = '{}.back'

    def __init__(self, population, dumpFilename):
        self.population = population
        self.dumpFilename = Path(dumpFilename)
        self.dumpFilenameBackup = Path(self.BACKUP_TEMPLATE.format(dumpFilename))

    @staticmethod
    def load(ID, tag, population):
        dumpFilename = Path(PopulationDump.DUMPFILE_TEMPLATE.format(ID, tag))
        if dumpFilename.exists():
            with open(dumpFilename, 'rb') as f:
                systems = pcl.load(f)
            if not set(systems.keys()) <= set(system['ID'] for system in population):
                dumpFilename.unlink()
                systems = {}
        else:
            systems = {}
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
