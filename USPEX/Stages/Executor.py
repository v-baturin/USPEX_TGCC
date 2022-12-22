"""
USPEX.Stages.Executor
=====================

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>

"""

import logging
import os, shutil
import asyncio
from os.path import join as pj

from .Connector import Connector

logger = logging.getLogger(__name__)


class Executor(object):
    CALC_FOLDER_TEMPLATE = 'CalcFold{}_{}'   # Path to folder to made QM/MM calculation

    _DEFAULT_SLEEP_TIME = 1
    _ATTEMPTS = 2

    knownInterfaces = {}
    knownTaskManagers = {}

    @classmethod
    def registerInterface(cls, name, interfaceType: type):
        assert name not in cls.knownInterfaces
        cls.knownInterfaces[name] = interfaceType

    @classmethod
    def registerTaskManager(cls, name, taskManagerType: type):
        assert name not in cls.knownTaskManagers
        cls.knownTaskManagers[name] = taskManagerType

    def __init__(self, type: str, commandExecutable: str, tag: str, workingDirectory: str = '.', remote=None,
                 taskManager=None, gather: bool = False, keepFolders: bool = False, sleepTime: int = None, **kwargs):
        """

        :param commandExecutable:
        :param workingDirectory:
        :param type:
        :param remote:
        :param taskManager:
        :type gather: bool
        :param gather: If True we will gather data needed for calculation replay during the run.
        """
        logger.debug('Created calculator.')

        self.commandExecutable = commandExecutable
        self.workingDirectory = workingDirectory
        self.tag = tag
        self.gather = gather
        self.keepFolders = keepFolders

        self._interface = self.knownInterfaces[type](tag, **kwargs)

        if sleepTime is not None and sleepTime > 0:
            self.sleepTime = sleepTime
        elif hasattr(self._interface, 'DEFAULT_SLEEP_TIME'):
            self.sleepTime = self._interface.DEFAULT_SLEEP_TIME
        else:
            self.sleepTime = self._DEFAULT_SLEEP_TIME

        if remote is None:
            remote = {}
        self._connector = Connector(**remote)

        if taskManager is not None:
            self._taskManager = self.knownTaskManagers[taskManager['type']](header=taskManager['header'],
                                                                            connector=self._connector)
        else:
            self._taskManager = self.knownTaskManagers['SHELL'](connector=self._connector)

        self.submittedTasks = {}

    async def run(self, system):
        ID = system['ID']
        tag = self.tag
        self._gatherSystems(system, tag, ioType='input')
        calcFolder = pj(self.workingDirectory, self.CALC_FOLDER_TEMPLATE.format(ID, tag))
        for attempt in range(self._ATTEMPTS):
            if calcFolder in self.submittedTasks:
                jobID = self.submittedTasks[calcFolder]
                logger.info(f'System {ID} with tag {tag} was already submitted as {jobID} job.')
            else:
                shutil.rmtree(calcFolder, ignore_errors=True)
                os.makedirs(calcFolder)
                args = self._interface.prepareLocalCalculation(system, calcFolder)
                self._gatherData(calcFolder, ioType='input')
                await self._connector.sync_l2r(calcFolder)
                logger.info(f'System {ID} with tag {tag} will be submitted now.')
                jobID = await self._taskManager.submit(f'{self.commandExecutable} {args}', f'USPEX-{ID}S{tag}',
                                                       self._interface.inputFile, self._interface.outputFile,
                                                       self._interface.errorFile, calcFolder)
                self.submittedTasks[calcFolder] = jobID

            if jobID > 0:
                while not await self._taskManager.isReady(jobID):
                    logger.debug(f'system {ID} will wait for update {self.sleepTime}s')
                    await asyncio.sleep(self.sleepTime)

            await self._connector.sync_r2l(calcFolder)
            await self._connector.clean(calcFolder)
            del self.submittedTasks[calcFolder]
            self._gatherData(calcFolder, ioType='output')

            if self._interface.isConverged(calcFolder):
                logger.debug('System converged. Proceeding update.')
                self._interface.readOutput(system, calcFolder)
                logger.info(f'system {ID} with tag {tag} relaxation successful.')
                if not self.keepFolders:
                    shutil.rmtree(calcFolder, ignore_errors=True)
                break
        else:
            raise RuntimeError(f'Task failed {self._ATTEMPTS} times')
        self._gatherSystems(system, tag, ioType='output')

    def _gatherSystems(self, system, tag : str, ioType : str):
        if self.gather:
            folder = pj(self.workingDirectory, 'GatheredData', ioType)
            os.makedirs(folder, exist_ok=True)
            from ..components import AtomisticRepresentation
            with open(pj(folder, f"system{system['ID']}_{tag}"), 'wt') as f:
                AtomisticRepresentation.writeAtomicStructure(f, system)

    def _gatherData(self, calcFolder : str, ioType : str):
        if self.gather:
            copytree(calcFolder, pj(self.workingDirectory, 'GatheredData', ioType, os.path.basename(calcFolder)))


def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)

