import logging
logger = logging.getLogger(__name__)

import os
import sys
import asyncio
import pickle as pcl
from shutil import copyfile
from copy import copy, deepcopy
from enum import Enum

from .Calculators.LifeState import LifeState
from .Calculators.Common.SHELL_Calculator import SHELL_Calculator, ReferenceMismatch
from .IO.OutputRepresentation import OutputRepresentation
from .IO.InputParser import read
from .IO.compileParams import compileParams


class ControllerState(Enum):
    createPopulation = 0
    processPopulation = 1
    updateOptimizer = 2
    runControllerLogic = 3


class GenerationController(object):

    INPUT_FILENAME = 'input.uspex'
    DUMP_FILENAME = "controller.dump"
    DUMP_FILENAME_BACKUP = "controller.dump.back"
    knownOptimizers = {}

    def __init__(self, numGenerations : int, stopCrit : int, numParallelCalcs : int, stages : list,
                 optimizer, outputRepresentation):
        self.numGenerations = numGenerations
        self.stopCrit = stopCrit
        self.numParallelCalcs = numParallelCalcs
        self.optimizer = optimizer
        self.stages = stages
        self.outputRepresentation = outputRepresentation
        self.doPresentSystems = True
        self.generation = 0
        self.numberStableGenerations = 0
        self.state = ControllerState.createPopulation
        self.population = None

        self.populations = []
        self.optimizers = []
        self.systems = {}

        self.save()

    async def run(self):
        self.outputRepresentation.presentOutput(self.populations, self.optimizers, self.optimizer)
        while (self.generation < self.numGenerations and
               self.numberStableGenerations < self.stopCrit and
               not self.optimizer.isGoalReached):

            if self.state is ControllerState.createPopulation:
                self.population, *info = self.optimizer.run()
                self.outputRepresentation.presentOutput(self.populations, self.optimizers, self.optimizer)
                self.state = ControllerState.processPopulation
                self.save()
            if self.state is ControllerState.processPopulation:
                lifeState = LifeState.load(self.population)
                sem = asyncio.Semaphore(self.numParallelCalcs)
                self.doPresentSystems = True
                task = asyncio.ensure_future(self.presentSystems())
                await asyncio.gather(*(self.life(lifeState, system, sem) for system in self.population))
                self.doPresentSystems = False
                await asyncio.wait({task})
                self.populations.append(copy(self.population))
                self.outputRepresentation.presentOutput(self.populations, self.optimizers, self.optimizer)
                self.state = ControllerState.updateOptimizer
                self.save()
            if self.state is ControllerState.updateOptimizer:
                self.optimizer.update(self.population)
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
        with open('USPEX_IS_DONE', 'wt') as f:
            f.write('')
        logger.info('Calculation finished.')

    async def life(self, state, system, sem):
        await sem.acquire()
        ID = system['ID']
        system['isBad'] = False
        if ID not in state.systems:
            state.systems[ID] = [copy(system)]
        processedSystems = state.systems[ID]
        self.systems[ID] = [deepcopy(system)]
        for i, stage in enumerate(self.stages):
            if i + 1 < len(processedSystems):
                system.update(processedSystems[i + 1])
            else:
                try:
                    await stage.run(system)
                except ReferenceMismatch:
                    exc_info = sys.exc_info()
                    raise exc_info[0].with_traceback(exc_info[1], exc_info[2])
                except Exception as ex:
                    logger.warning(f'system {ID} error in relaxation:')
                    logger.exception(ex)
                    system['isBad'] = True
                    break
                processedSystems.append(copy(system))
            self.systems[ID].append(deepcopy(system))
            # TODO this is ugly workaround
            # if not system['structure'].isGoodSystem():
            #     logger.info(f'system {ID} violates constraints')
            #     system['isBad'] = True
            #     break

            state.save()

        sem.release()

    async def presentSystems(self):
        while self.doPresentSystems:
            await asyncio.sleep(120)
            self.outputRepresentation.presentSystems(self.systems, self.optimizer)

    def save(self):
        if os.path.exists(GenerationController.DUMP_FILENAME):
            copyfile(GenerationController.DUMP_FILENAME, GenerationController.DUMP_FILENAME_BACKUP)
        with open(GenerationController.DUMP_FILENAME, 'wb') as f:
            pcl.dump(self, f)

    @staticmethod
    def createController():
        if os.path.exists(GenerationController.DUMP_FILENAME):
            with open(GenerationController.DUMP_FILENAME, 'rb') as f:
                controller = pcl.load(f)
            logger.info('Calculation initialized from dump file.')
        elif os.path.exists(GenerationController.INPUT_FILENAME):
            input = read(GenerationController.INPUT_FILENAME)
            params = compileParams(**input)
            optimizer = params['optimizer']
            stages = params['stages']
            numParallelCalcs = params['numParallelCalcs']
            numGenerations = params['numGenerations']
            stopCrit = params['stopCrit']

            if optimizer['type'] in GenerationController.knownOptimizers:
                optimizer = GenerationController.knownOptimizers[optimizer['type']](**optimizer)
            else:
                RuntimeError(f"Unknown optimizer type: {optimizer['type']}.")
            stages = [SHELL_Calculator(**stage) for stage in stages]
            outputRepresentation = OutputRepresentation(optimizer, **params)
            controller = GenerationController(numGenerations, stopCrit, numParallelCalcs, stages, optimizer,
                                              outputRepresentation)
            logger.info('Calculation initialized from input parameters.')
        else:
            raise RuntimeError('No input or dump file to start.')
        return controller

    @staticmethod
    def registerOptimizer(optimizerType: type):
        assert optimizerType.__name__ not in GenerationController.knownOptimizers
        GenerationController.knownOptimizers[optimizerType.__name__] = optimizerType