"""
USPEX.Stages.Executor
=====================

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>

"""

import asyncio
import asyncio
import logging
import shutil

from pathlib import Path

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

    def __init__(self, type: str,
                       commandExecutable: str,
                       tag: str,
                       workingDirectory: str = '.',
                       remote=None,
                       taskManager=None,
                       gather: bool = False,
                       keepFolders: bool = False,
                       sleepTime: int = None,
                       **kwargs):
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
        self.workingDirectory = Path(workingDirectory)
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
        calcFolder = self.workingDirectory/self.CALC_FOLDER_TEMPLATE.format(ID, tag)
        for attempt in range(self._ATTEMPTS):
            if calcFolder in self.submittedTasks:
                jobID = self.submittedTasks[calcFolder]
                logger.info(f'System {ID} with tag {tag} was already submitted as {jobID} job.')
            else:
                shutil.rmtree(calcFolder, ignore_errors=True)
                calcFolder.mkdir(parents=True)
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
                results = self._interface.readOutput(system, calcFolder)
                logger.info(f'system {ID} with tag {tag} relaxation successful.')
                if not self.keepFolders:
                    shutil.rmtree(calcFolder, ignore_errors=True)
                break
        else:
            raise RuntimeError(f'Task failed {self._ATTEMPTS} times')
        self._gatherSystems(system, tag, ioType='output')
        return results

    def _gatherSystems(self, system, tag: str, ioType: str):
        if self.gather:
            folder = self.workingDirectory/'GatheredData'/ioType
            folder.mkdir(exist_ok=True)
            from ..components import AtomisticRepresentation
            with open(folder/f"system{system['ID']}_{tag}", 'wt') as f:
                AtomisticRepresentation.writeAtomicStructure(f, system)

    def _gatherData(self, calcFolder: Path, ioType : str):
        if self.gather:
            copytree(calcFolder, self.workingDirectory/'GatheredData'/ioType/calcFolder.name)


def copytree(src: Path, dst: Path, symlinks=False, ignore=None):
    dst.mkdir(exist_ok=True, parents=True)
    for item in src.iterdir():
        d = dst/item.name
        if item.is_dir():
            shutil.copytree(item, d, symlinks, ignore)
        else:
            if not d.exists() or item.stat().st_mtime - d.stat().st_mtime > 1:
                shutil.copy2(item, d)
