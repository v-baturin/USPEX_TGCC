"""
USPEX.Stages.TaskManagers.SBATCH
================================

"""

import logging

from pathlib import Path

logger = logging.getLogger(__name__)


class SBATCH:
    '''

    '''

    type = 'SBATCH'
    _RUNSCRIPT = 'jobscript'

    def __init__(self, header : str, connector):
        self.header = header
        self.connector = connector

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

        content = ''
        for line in self.header.split('\n'):
            if ' -j ' in line.lower():
                logger.info('Job name found in HEADER will be overwritten')
            elif ' -i ' in line.lower():
                logger.info('Input file name found in HEADER will be overwritten')
            elif ' -o ' in line.lower():
                logger.info('Output file name found in HEADER will be overwritten')
            elif ' -e ' in line.lower():
                logger.info('Error file name found in HEADER will be overwritten')
            else:
                content += line + '\n'
        content += f'#SBATCH -J  {JOB_NAME}\n' \
                   f'#SBATCH -i  {inputFile}\n' \
                   f'#SBATCH -o  {outputFile}\n' \
                   f'#SBATCH -e  {errorFile}\n\n' \
                   f'{COMMAND_EXEC}\n'

        return ''.join(content)

    async def submit(self, command: str,
                           jobname: str,
                           input: str,
                           output: str,
                           error: str,
                           calcFolder: Path) -> int:
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

        returncode, out, err = await self.connector.execute(f'sbatch {self._RUNSCRIPT}', cwd=str(calcFolder))
        logger.debug(f'process returned code {returncode}')
        if returncode != 0:
            logger.error(err)
            logger.error(out)

        jobID = self._parseJobID(out, err)
        logger.info(f"Job ID is {jobID}")
        return jobID

    def _parseJobID(self, output : str, error : str) -> int:
        '''
        :param output: output message
        :param error: error message
        :return: jobID
        '''

        return int(output.split()[-1])

    async def isReady(self, jobID : int):
        returncode, out, err = await self.connector.execute(f'squeue -t all -j {jobID}')
        if 'invalid job' in err.lower():
            return True
        header, job, *_ = out.split('\n')
        i = header.split().index('ST')
        status = job.split()[i]
        return status == 'CD' or status == 'F' or status == 'CA' or status == 'S'

    async def isExist(self, jobID : int):
        returncode, out, err = await self.connector.execute(f'squeue -j {jobID}')
        header, job, *_ = out.split('\n')
        i = header.split().index('ST')
        status = job.split()[i]
        return status == 'R' or status == 'PD'

    async def kill(self, jobID : int):
        returncode, out, err = await self.connector.execute(f'scancell {jobID}')

        #logger.info(f'Process with jobID={}  killed.')
