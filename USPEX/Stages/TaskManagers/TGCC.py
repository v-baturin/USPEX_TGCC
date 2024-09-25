"""
USPEX.Stages.TaskManagers.TGCC
================================

"""

import logging

from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TGCC:
    '''

    '''

    type = 'TGCC'
    _RUNSCRIPT = 'jobscript'
    _QUEUE_REFRESH_DELAY_SECONDS = 300

    def __init__(self, header : str, connector, refreshDelay : int = None):
        self.header = header
        self.connector = connector
        self.jobsStatusCache = dict()
        self.queueRefreshDelay = self._QUEUE_REFRESH_DELAY_SECONDS if refreshDelay is None else refreshDelay
        self.cacheTime = datetime.now()

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'queueRefreshDelay'): # TODO: remove after testing
            self.queueRefreshDelay = self._QUEUE_REFRESH_DELAY_SECONDS
        self.updateCache()

    def _prepareSubmission(self, COMMAND_EXEC : str,
                                 JOB_NAME : str,
                                 inputFile : str,
                                 outputFile : str,
                                 errorFile : str) -> str:
        '''
        Preparing jobscript for submission
        :param commandExec:
        :param jobName:

        :param COMMAND_EXEC: command executable
        :param JOB_NAME:
        :param inputFile: path to inputFile
        :param outputFile: path to outputFile
        :param errorFile: path to errorFile
        :return: jobscript as string
        '''
        hash_lines = []
        non_hash_lines = []
        content = ''
        for line in self.header.split('\n'):
            if not line:
                continue
            if line.strip()[0] == '#':
                if ' -r ' in line.lower():
                    logger.info('Job name found in HEADER will be overwritten')
                elif ' -o ' in line.lower():
                    logger.info('Output file name found in HEADER will be overwritten')
                elif ' -e ' in line.lower():
                    logger.info('Error file name found in HEADER will be overwritten')
                    content += f'#MSUB  -e  {errorFile}\n'
                else:
                    hash_lines.append(line)
            else:
                non_hash_lines.append(line)

        content += '\n'.join(hash_lines)
        content += f'\n#MSUB -r  {JOB_NAME}\n' \
                   f'#MSUB  -o  {outputFile}\n' \
                   f'#MSUB  -e  {errorFile}\n' \

        content += '\n'.join(non_hash_lines) + f'\n{COMMAND_EXEC}\n'

        return ''.join(content)

    async def submit(self, command: str, jobname: str, input: str, output: str, error: str, calcFolder: Path) -> int:
        '''
        :param command: command executable
        :param jobname: name of the job
        :param input: input file path
        :param output: output file path
        :param error: error file path
        :param calcFolder: path to the calcFolder directory
        :return:
        '''
        content = self._prepareSubmission(command, jobname, input, output, error)
        filepath = calcFolder/self._RUNSCRIPT
        with open(filepath, 'wt') as f:
            f.write(content)
        await self.connector.sync_l2r(filepath)
        logger.debug(f'Trying to submit a task in {calcFolder}')
        returncode, out, err = await self.connector.execute(f'ccc_msub {self._RUNSCRIPT}', cwd=calcFolder)
        logger.debug(f'process returned code {returncode}')
        if returncode != 0:
            logger.error(err)
            logger.error(out)

        jobID = self._parseJobID(out, err)
        logger.info(f"Job in {calcFolder}. ID = {jobID}")
        self.jobsStatusCache[jobID] = 'PD'
        return jobID

    def _parseJobID(self, output : str, error : str) -> int:
        '''
        :param output: output message
        :param error: error message
        :return: jobID
        '''

        return int(output.split()[-1])

    async def updateCache(self):
        # Execute the command to get all jobs status
        returncode, out, err = await self.connector.execute('squeue -t all -u $USER')

        # Parse the output and update the cache
        lines = out.split('\n')
        header = lines[0].split()
        status_index = header.index('ST')
        job_id_index = header.index('JOBID')

        new_cache = dict()
        if len(lines) > 1:
            for line in lines[1:]:
                if line.strip():
                    parts = line.split()
                    job_id = int(parts[job_id_index])
                    status = parts[status_index]
                    new_cache[job_id] = status

        self.jobsStatusCache = new_cache
        self.cacheTime = datetime.now()

    async def ensureCacheIsFresh(self):
        # Check if the cache is older than self.queueRefreshDelay seconds and update if necessary
        if datetime.now() - self.cacheTime > timedelta(seconds=self.queueRefreshDelay):
            await self.updateCache()

    async def isReady(self, jobID: int):
        await self.ensureCacheIsFresh()
        status = self.jobsStatusCache.get(jobID, False)
        return status in {'CD', 'F', 'CA', 'S', False}

    async def isExist(self, jobID: int):
        await self.ensureCacheIsFresh()
        status = self.jobsStatusCache.get(jobID, False)
        return status in {'R', 'PD'} if status else status

    async def kill(self, jobID : int):
        returncode, out, err = await self.connector.execute(f'ccc_mdel {jobID}')

        #logger.info(f'Process with jobID={}  killed.')
