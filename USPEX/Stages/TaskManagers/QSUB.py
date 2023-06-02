"""
USPEX.Stages.TaskManagers.QSUB
==============================

"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class QSUB:

    shortname = 'QSUB'
    _RUNSCRIPT = 'jobscript'


    def __init__(self, header : str, connector):
        """
        :param header: description of params of TaskManager
        :param connector: for remote submission
        """

        self.connector = connector
        self.header = header

    def _prepareSubmission(self, COMMAND_EXEC: str,
                                 JOB_NAME: str,
                                 inputFile: str,
                                 outputFile : str,
                                 errorFile : str) -> str:
        """
        Preparing jobscript for submission
        :param commandExec:
        :param jobName:
        :return: jobscript as string
        """

        content = ''
        for line in self.header.split('\n'):
            if ' -N ' in line:
                logger.info('Job name found in HEADER will be overwritten')
            elif ' -o ' in line:
                logger.info('Output file name found in HEADER will be overwritten')
            elif ' -e ' in line:
                logger.info('Error file name found in HEADER will be overwritten')
            else:
                content += line + '\n'
        content += f'#PBS -N  {JOB_NAME}\n'
        content += f'#PBS -o  {outputFile}\n'
        content += f'#PBS -e  {errorFile}\n'
        content += '\ncd "$PBS_O_WORKDIR"\n'
        content += f'\n {COMMAND_EXEC}\n\n'

        return ''.join(content)

    async def submit(self, command: str,
                           jobname: str,
                           input: str,
                           output: str,
                           error: str,
                           calcFolder: Path) -> int:
        content = self._prepareSubmission(command, jobname, input, output, error)
        with open(calcFolder/self._RUNSCRIPT, 'wt') as f:
            f.write(content)
        await self.connector.sync_l2r(calcFolder/self._RUNSCRIPT)

        returncode, out, err = await self.connector.execute(f'qsub {self._RUNSCRIPT}', cwd=str(calcFolder))

        logger.debug(f'process returned code {returncode}')
        if returncode != 0:
            logger.error(err)
            logger.error(out)

        jobID = self._parseJobID(out, err)
        logger.info(f"Job ID is {jobID}")
        return jobID

    def _parseJobID(self, output : str, error : str) -> int:
        """

        :param output:
        :param error:
        :return: jobID
        """

        if '.mgmt' in output:
            tmp = output.index('.mgmt')
            return int(output[:tmp])
        elif 'job' in output:
            tmp = output.index('job') + 4
            return int(output[tmp:])
        else:
            return int(output)

    async def isReady(self, jobID):
        returncode, out, err = await self.connector.execute(f'qstat {jobID}')
        return not (' R ' in out or ' Q ' in out)

    async def isExist(self, jobID):
        returncode, out, err = await self.connector.execute(f'qstat {jobID}')
        return len(out) > 0

    async def kill(self, jobID):
        returncode, out, err = await self.connector.execute(f'qdel {jobID}')
        #logger.info(f'Process with jobID={jobID}  killed.')


