"""
USPEX.Calculators.Common.SHELL_Calculator
=========================================

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>

"""

import logging
import os, shutil
import asyncio
import filecmp
from os.path import join as pj

from .Connector import Connector

logger = logging.getLogger(__name__)


class SHELL_Calculator(object):
    CALC_FOLDER_TEMPLATE = 'CalcFold{}_{}'   # Path to folder to made QM/MM calculation

    _DEFAULT_SLEEP_TIME = 1


    def __init__(self, type : str, commandExecutable : str, tag : str, workingDirectory : str = '.',
                 remote=None, taskManager=None, gather : bool = False, referenceFolder : str = None,
                 keepFolders : bool = False, sleepTime : int = None, **kwargs):
        """

        :param commandExecutable:
        :param workingDirectory:
        :param type:
        :param remote:
        :param taskManager:
        :type gather: bool
        :param gather: If True we will gather data needed for calculation replay during the run.
        :type referenceFolder: str
        :param referenceFolder: If specified we will emulate run of actual third party program by copying data from
        reference folder and compare resulting systems with stored in reference folder. If not specified do normal run.
        :param kwargs:
        """
        logger.debug('Created calculator.')

        self.commandExecutable = commandExecutable
        self.workingDirectory = workingDirectory

        self._ATTEMPTS = 2

        if type == 'gulp':
            from USPEX.Calculators.Interfaces.GULP_Interface import GULP_Interface
            self._interface = GULP_Interface(tag, **kwargs)
        elif type == 'vasp':
            from USPEX.Calculators.Interfaces.VASP_Interface import VASP_Interface
            self._interface = VASP_Interface(tag, **kwargs)
        elif type == 'abinit':
            from USPEX.Calculators.Interfaces.ABINIT_Interface import ABINIT_Interface
            self._interface = ABINIT_Interface(tag, **kwargs)
        elif type == 'lammps':
            from USPEX.Calculators.Interfaces.LAMMPS_Interface import LAMMPS_Interface
            self._interface = LAMMPS_Interface(tag, **kwargs)
        elif type == 'qe':
            from USPEX.Calculators.Interfaces.QE_Interface import QE_Interface
            self._interface = QE_Interface(tag, **kwargs)
        elif type == 'mlip':
            from USPEX.Calculators.Interfaces.MLIP_Interface import MLIP_Interface
            self._interface = MLIP_Interface(tag, **kwargs)
        elif type == 'mopac':
            from USPEX.Calculators.Interfaces.MOPAC_Interface import MOPAC_Interface
            self._interface = MOPAC_Interface(tag, **kwargs)
        elif type == 'aims':
            from USPEX.Calculators.Interfaces.FHIaims_Interface import FHIaims_Interface
            self._interface = FHIaims_Interface(tag, **kwargs)
        else:
            raise RuntimeError(f'Invalid Calculator Type: {type}')

        if sleepTime is not None and sleepTime > 0:
            self.sleepTime = sleepTime
        elif hasattr(self._interface, 'DEFAULT_SLEEP_TIME'):
            self.sleepTime = self._interface.DEFAULT_SLEEP_TIME
        else:
            self.sleepTime = self._DEFAULT_SLEEP_TIME

        self.tag = tag

        if remote is None:
            remote = {}
        self._connector = Connector(**remote)

        if taskManager is not None:
            if taskManager['type'] == 'BSUB':
                from .TaskManagers.BSUB import BSUB
                self._taskManager = BSUB(taskManager['header'], connector=self._connector)
            elif taskManager['type'] == 'QSUB':
                from .TaskManagers.QSUB import QSUB
                self._taskManager = QSUB(taskManager['header'], connector=self._connector)
            elif taskManager['type'] == 'SBATCH':
                from .TaskManagers.SBATCH import SBATCH
                self._taskManager = SBATCH(taskManager['header'], connector=self._connector)
            else:
                raise RuntimeError(f'Invalid Task Manager Type: {taskManager["type"]}')
        else:
            from .TaskManagers.SHELL import SHELL
            self._taskManager = SHELL(connector=self._connector)
            logger.debug('     with a task manager')

        self.submittedTasks = {}
        self.gather = gather
        if self.gather:
            self.gatheredDataPath = pj(self.workingDirectory, 'GatheredData')

        if referenceFolder is not None:
            self.referenceFolder = pj(self.workingDirectory, referenceFolder)
        else:
            self.referenceFolder = None

        self.keepFolders = keepFolders

    async def run(self, system):
        ID = system['ID']
        tag = self.tag
        calcFolder = pj(self.workingDirectory, self.CALC_FOLDER_TEMPLATE.format(ID, tag))
        for attempt in range(self._ATTEMPTS):
            if calcFolder in self.submittedTasks:
                jobID = self.submittedTasks[calcFolder]
                logger.info(f'System {ID} with tag {tag} was already submitted as {jobID} job.')
            else:
                shutil.rmtree(calcFolder, ignore_errors=True)
                os.makedirs(calcFolder)
                self._interface.prepareLocalCalculation(system, calcFolder)
                if self.referenceFolder is not None:
                    self._compareWithReference(system, tag, calcFolder, ioType='input')
                if self.gather:
                    self._gatherData(system, tag, calcFolder, ioType='input')
                await self._connector.sync_l2r(calcFolder)
                logger.info(f'System {ID} with tag {tag} will be submitted now.')
                jobID = await self._taskManager.submit(self.commandExecutable, f'USPEX-{ID}S{tag}',
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

            if self.referenceFolder is not None:
                self._pickUpFromReference(ID, tag, calcFolder)

            if self._interface.isConverged(calcFolder):
                logger.debug('System converged. Proceeding update.')
                self._interface.readOutput(system, calcFolder)
                if self.referenceFolder is not None:
                    self._compareWithReference(system, tag, calcFolder, ioType='output')
                if self.gather:
                    self._gatherData(system, tag, calcFolder, ioType ='output')
                logger.info(f'system {ID} with tag {tag} relaxation successful.')
                if not self.keepFolders:
                    shutil.rmtree(calcFolder, ignore_errors=True)
                return
            else:
                if self.gather:
                    self._gatherData(system, tag, calcFolder, ioType ='output')

        raise RuntimeError(f'Task failed {self._ATTEMPTS} times')

    def _compareWithReference(self, system, tag : str, calcFolder : str, ioType : str):
        if ioType == 'input':
            folder = pj(self.referenceFolder, 'input')
            calcFolderRef = pj(folder, self.CALC_FOLDER_TEMPLATE.format(system['ID'], tag))
            dcmp = filecmp.dircmp(calcFolderRef, calcFolder)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            if not match:
                raise ReferenceMismatch(f"input mismatch in {system['ID']}, with tag {tag}.")
        elif ioType == 'output':
            folder = pj(self.referenceFolder, 'output')
            with open(pj(folder, f"system{system['ID']}_{tag}"), 'rt') as f:
                systemRef = system['structure'].fromJSON(f.read())
            system['structure']._fingerprint = None
            systemRef._fingerprint = None
            if not system['structure'] == systemRef:
                raise ReferenceMismatch(f"output mismatch in {system['ID']}, with tag {tag}. "
                                        f"Ref system is {systemRef.toJSON()}, actual system is {system['structure'].toJSON()}")

    def _pickUpFromReference(self, ID : int, tag : str, calcFolder : str):
        folder = pj(self.referenceFolder, 'output')
        calcFolderRef = pj(folder, self.CALC_FOLDER_TEMPLATE.format(ID, tag))
        shutil.rmtree(calcFolder, ignore_errors=True)
        shutil.copytree(calcFolderRef, calcFolder)

    def _gatherData(self, system, tag : str, calcFolder : str, ioType : str):
        if not (ioType == 'input' or ioType == 'output'):
            return
        folder = pj(self.gatheredDataPath, ioType)
        copytree(calcFolder, pj(folder, os.path.basename(calcFolder)))
        from ..components import AtomisticRepresentation
        with open(pj(folder, f"system{system['ID']}_{tag}"), 'wt') as f:
            AtomisticRepresentation.writeAtomicStructure(f, system)


class ReferenceMismatch(Exception):
    pass


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

