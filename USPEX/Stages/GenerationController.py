import logging

import time

import asyncio
import pickle as pcl

from copy import copy, deepcopy
from enum import Enum
from pathlib import Path
from shutil import copyfile
from time import time
from ..IO.OutputRepresentation import OutputRepresentation
from ..IO.InputParser import read


logger = logging.getLogger(__name__)
DEFAULT_OUTPUT_REFRESH_DELAY = 120
DEFAULT_EXECUTION_TIME = 86400  # 24h run

class ControllerState(Enum):
    createPopulation = 0
    processPopulation = 1
    updateOptimizer = 2
    runControllerLogic = 3


class GenerationController(object):

    INPUT_FILENAME = Path('input.uspex')
    DUMP_FILENAME = Path("controller.dump")
    DUMP_FILENAME_BACKUP = Path("controller.dump.back")
    knownOptimizers = {}
    populationProcessorType = None
    compileParams = None

    @classmethod
    def registerOptimizer(cls, optimizerType: type):
        assert optimizerType.__name__ not in cls.knownOptimizers
        cls.knownOptimizers[optimizerType.__name__] = optimizerType

    @classmethod
    def setPopulationProcessor(cls, populationProcessorType):
        cls.populationProcessorType = populationProcessorType

    @classmethod
    def setUpcompileParams(cls, compileParams):
        cls.compileParams = compileParams

    def __init__(self, numGenerations : int, stopCrit : int, numParallelCalcs : int, stages : list,
                 optimizer, outputRepresentation, outputRefreshDelay, executionTime, start):
        self.numGenerations = numGenerations
        self.stopCrit = stopCrit
        self.optimizer = optimizer
        self.outputRepresentation = outputRepresentation
        self.outputRefreshDelay = outputRefreshDelay
        self.start = start
        self.executionTime = executionTime
        self.doPresentSystems = True
        self.generation = 0
        self.numberStableGenerations = 0
        self.state = ControllerState.createPopulation
        self.population = None
        self.populations = []
        self.optimizers = []
        self.systems = {}
        self.populationProcessor = self.populationProcessorType(tag='stages', stages=stages,
                                                                inputKey='population',
                                                                numParallelCalcs=numParallelCalcs,
                                                                systems=self.systems,
                                                                checkCallback=self.optimizer.target.constraints.systemCheckAndFix)
        self.save()

    @staticmethod
    def createController(start):
        if GenerationController.DUMP_FILENAME.exists():
            with open(GenerationController.DUMP_FILENAME, 'rb') as f:
                controller = pcl.load(f)
                controller.start = start
            logger.info('Calculation initialized from dump file.')
        elif GenerationController.INPUT_FILENAME.exists():
            params = GenerationController.compileParams(read(GenerationController.INPUT_FILENAME))
            optimizer = params['optimizer']
            numParallelCalcs = params['numParallelCalcs']
            numGenerations = params['numGenerations']
            stopCrit = params['stopCrit']
            outputRefreshDelay = params['outputRefreshDelay'] if 'outputRefreshDelay' in params \
                else DEFAULT_OUTPUT_REFRESH_DELAY
            executionTime = params['executionTime']if 'executionTime' in params \
                else DEFAULT_EXECUTION_TIME
            if optimizer['type'] in GenerationController.knownOptimizers:
                optimizer = GenerationController.knownOptimizers[optimizer['type']](**optimizer)
            else:
                RuntimeError(f"Unknown optimizer type: {optimizer['type']}.")
            stages = params['stages']
            outputRepresentation = OutputRepresentation(optimizer, **params)
            controller = GenerationController(numGenerations, stopCrit, numParallelCalcs, stages, optimizer,
                                              outputRepresentation, outputRefreshDelay, executionTime, start)
            logger.info('Calculation initialized from input parameters.')
        else:
            raise RuntimeError('No input or dump file to start.')
        return controller

    async def run(self):
        # self.outputRepresentation.presentOutput(self.populations, self.optimizers, self.optimizer)
        while (self.generation < self.numGenerations and
               self.numberStableGenerations < self.stopCrit and
               not self.optimizer.isGoalReached):

            if self.state is ControllerState.createPopulation:
                self.population = self.optimizer.createPopulation()
                self.outputRepresentation.presentOutput(self.populations, self.optimizers, self.optimizer)
                self.state = ControllerState.processPopulation
                self.save()
            if self.state is ControllerState.processPopulation:
                self.doPresentSystems = True
                task = asyncio.ensure_future(self.presentSystems())
                system = await self.populationProcessor.run(dict(ID='USPEX', population=self.population))
                self.population = system['population']
                self.doPresentSystems = False
                await asyncio.wait({task})
                self.populations.append(copy(self.population))
                # self.outputRepresentation.presentOutput(self.populations, self.optimizers, self.optimizer)
                self.state = ControllerState.updateOptimizer
                self.save()
            if self.state is ControllerState.updateOptimizer:
                await self.optimizer.update(self.population)
                self.optimizers.append(copy(self.optimizer))
                self.outputRepresentation.presentOutput(self.populations, self.optimizers, self.optimizer)
                self.state = ControllerState.runControllerLogic
                self.save()
            if self.state is ControllerState.runControllerLogic:
                self.generation += 1
                if self.optimizer.isStable:
                    self.numberStableGenerations += 1
                else:
                    self.numberStableGenerations = 0
                self.state = ControllerState.createPopulation
                self.save()
        self.outputRepresentation.presentOutput(self.populations, self.optimizers, self.optimizer, final=True)
        with open('USPEX_IS_DONE', 'wt') as f:
            f.write('')
        logger.info('Calculation finished.')

    async def presentSystems(self):
        n = self.outputRefreshDelay
        while self.doPresentSystems:
            n -= 1
            if n < 0:
                self.outputRepresentation.presentSystems(self.systems, self.optimizer)
                n = self.outputRefreshDelay
            await asyncio.sleep(1)
        self.outputRepresentation.presentSystems(self.systems, self.optimizer)

    def save(self):
        if GenerationController.DUMP_FILENAME.exists():
            copyfile(GenerationController.DUMP_FILENAME, GenerationController.DUMP_FILENAME_BACKUP)
        with open(GenerationController.DUMP_FILENAME, 'wb') as f:
            pcl.dump(self, f)
        dt = time() - self.start
        logger.debug(f"{int(dt)} seconds passed")
        if dt >= self.executionTime:
            logger.info("Time is up. Exiting")
            exit()
