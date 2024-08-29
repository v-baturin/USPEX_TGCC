import logging

import asyncio
import pickle as pcl

from copy import copy, deepcopy
from enum import Enum
from pathlib import Path
from shutil import copyfile
from time import time
from ..IO.OutputRepresentation import OutputRepresentation
from ..IO.InputParser import read
from ..Semantics.Generator import Generator as GeneratorSemantics
from ..Semantics.Optimizer import Optimizer as OptimizerSemantics


logger = logging.getLogger(__name__)
DEFAULT_OUTPUT_REFRESH_DELAY = 120
DEFAULT_EXECUTION_TIME = None  # time in sec. None=infinite run

class ControllerState(Enum):
    createPopulation = 0
    processPopulation = 1
    updateOptimizer = 2
    runControllerLogic = 3


class GenerationController(object):

    INPUT_FILENAME = Path('input.yaml')
    DUMP_FILENAME = Path("controller.dump")
    DUMP_FILENAME_BACKUP = Path("controller.dump.back")
    knownOptimizers = {}
    knownGenerators = {}
    populationProcessorType = None
    compileParams = None

    @classmethod
    def registerOptimizer(cls, optimizerType: type):
        assert optimizerType.__name__ not in cls.knownOptimizers
        cls.knownOptimizers[optimizerType.__name__] = optimizerType

    @classmethod
    def registerGenerator(cls, generatorType: type):
        assert generatorType.__name__ not in cls.knownGenerators, f'{generatorType.__name__} is not a known Generator'
        cls.knownGenerators[generatorType.__name__] = generatorType

    @classmethod
    def setPopulationProcessor(cls, populationProcessorType):
        cls.populationProcessorType = populationProcessorType

    @classmethod
    def setUpcompileParams(cls, compileParams):
        cls.compileParams = compileParams

    def __init__(self, numGenerations : int, stopCrit : int, numParallelCalcs : int, stages: list,
                 optimizer, generator, outputRepresentation, outputRefreshDelay, executionTime, start):
        self.numGenerations = numGenerations
        self.stopCrit = stopCrit
        self.numParallelCalcs = numParallelCalcs
        self.stages = stages
        self.optimizer: OptimizerSemantics = optimizer
        self.generator: GeneratorSemantics = generator
        self.outputRepresentation = outputRepresentation
        self.outputRefreshDelay = outputRefreshDelay
        self.start = start
        self.executionTime = executionTime
        self.doPresentSystems = True
        self.generation = 0
        self.numberStableGenerations = 0
        self.isStable = False
        self.isGoalReached = False
        self.state = ControllerState.createPopulation
        self.population = None
        self.allSystems = None
        self.generations = self.generator.target.createGenerations()
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
            generator = params['generator']
            numParallelCalcs = params['numParallelCalcs']
            numGenerations = params['numGenerations']
            stopCrit = params['stopCrit']
            outputRefreshDelay = params['outputRefreshDelay'] if 'outputRefreshDelay' in params \
                else DEFAULT_OUTPUT_REFRESH_DELAY
            executionTime = params['executionTime'] if 'executionTime' in params \
                else DEFAULT_EXECUTION_TIME
            if optimizer['type'] in GenerationController.knownOptimizers:
                optimizer = GenerationController.knownOptimizers[optimizer['type']](**optimizer)
            else:
                RuntimeError(f"Unknown optimizer type: {optimizer['type']}.")
            generator = GenerationController.knownGenerators[generator['type']](**generator)

            stages = params['stages']
            outputRepresentation = OutputRepresentation(optimizer, generator, **params)
            controller = GenerationController(numGenerations, stopCrit, numParallelCalcs, stages, optimizer, generator,
                                              outputRepresentation, outputRefreshDelay, executionTime, start)
            logger.info('Calculation initialized from input parameters.')
        else:
            raise RuntimeError('No input or dump file to start.')
        return controller

    async def run(self, shellEnv=None):
        self.outputRepresentation.presentOutput(self.optimizer, self.generator, self.generations)
        while (self.generation < self.numGenerations and
               self.numberStableGenerations < self.stopCrit and
               not self.isGoalReached):

            if self.state is ControllerState.createPopulation:
                generation = self.generations[-1] if self.generations else None
                self.population = self.generator.call(generation)
                if generation is None:
                    self.allSystems = copy(self.population)
                else:
                    self.allSystems = copy(generation['allSystems'])
                    for ID in self.population.getIDs():
                        self.allSystems.addEntry(self.population.getEntry(ID))
                self.state = ControllerState.processPopulation
                self.save()
            if self.state is ControllerState.processPopulation:
                self.doPresentSystems = True
                task = asyncio.ensure_future(self.presentSystems())
                await self.populationProcessorType.processPopulation(self.stages, self.population,
                                                                     self.numParallelCalcs, self.generator.target,
                                                                     shellEnv=shellEnv)
                self.doPresentSystems = False
                await asyncio.wait({task})
                self.state = ControllerState.updateOptimizer
                self.save()
            if self.state is ControllerState.updateOptimizer:
                parents = self.generations[-1] if self.generations else None
                generation, self.isStable, self.isGoalReached = await self.optimizer.update(self.population, parents)
                if generation is not None:
                    self.generations.append(generation)
                self.outputRepresentation.presentOutput(self.optimizer, self.generator, self.generations)
                self.state = ControllerState.runControllerLogic
                self.save()
            if self.state is ControllerState.runControllerLogic:
                self.generation += 1
                if self.isStable:
                    self.numberStableGenerations += 1
                else:
                    self.numberStableGenerations = 0
                self.state = ControllerState.createPopulation
                self.save()
        self.outputRepresentation.presentOutput(self.optimizer, self.generator, self.generations, final=True)
        with open('USPEX_IS_DONE', 'wt') as f:
            f.write('')
        logger.info('Calculation finished.')

    async def presentSystems(self):
        n = self.outputRefreshDelay
        while self.doPresentSystems:
            n -= 1
            if n < 0:
                self.outputRepresentation.presentSystems(self.generations, self.allSystems)
                n = self.outputRefreshDelay
            await asyncio.sleep(1)
        self.outputRepresentation.presentSystems(self.generations, self.allSystems)

    def save(self):
        if GenerationController.DUMP_FILENAME.exists():
            copyfile(GenerationController.DUMP_FILENAME, GenerationController.DUMP_FILENAME_BACKUP)
            copyfile(GenerationController.DUMP_FILENAME.parent / 'uspex.db',
                     GenerationController.DUMP_FILENAME.parent / 'uspex.db.back')
        with open(GenerationController.DUMP_FILENAME, 'wb') as f:
            pcl.dump(self, f)
        if self.executionTime is not None:
            dt = time() - self.start
            logger.debug(f"{int(dt)} seconds passed")
            if dt >= self.executionTime:
                logger.info(f"Time is up. Exiting normally. Next state is {self.state.name}")
                exit()
