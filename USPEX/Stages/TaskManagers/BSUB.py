"""
USPEX.Stages.TaskManagers.BSUB
==============================

"""


import logging

logger = logging.getLogger(__name__)


class BSUB:

    shortname = 'BSUB'
    _RUNSCRIPT = 'jobscript'

    def __init__(self, header : str, connector):
        """

        :param header: description of params of TaskManager
        :param connector: for remote submission
        :return:
        """

        self.connector = connector
        self.header = header

    def _prepareSubmission(self, COMMAND_EXEC: str,
                                 JOB_NAME : str,
                                 inputFile : str,
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
        content += f'#BSUB -J  {JOB_NAME}\n'
        content += f'#BSUB -i  {inputFile}\n'
        content += f'#BSUB -o  {outputFile}\n'
        content += f'#BSUB -e  {errorFile}\n'
        content += f'\n {COMMAND_EXEC}\n\n'

        return ''.join(content)

    async def submit(self, command: str,
                           jobname: str,
                           input: str,
                           output: str,
                           error: str,
                           calcFolder: str) -> int:
        content = self._prepareSubmission(command, jobname, input, output, error)
        with open(calcFolder/self._RUNSCRIPT, 'wt') as f:
            f.write(content)
        await self.connector.sync_l2r(calcFolder/self._RUNSCRIPT)

        returncode, out, err = await self.connector.execute('bsub', cwd=str(calcFolder), input=content)
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

        if 'Job <' in output:
            tmpA = output.index('Job <') + 5
            tmpB = output.index('>',tmpA)
            return int(output[tmpA: tmpB])
        return -1

    async def isReady(self, jobID):
        returncode, out, err = await self.connector.execute(f'bjobs {jobID}')
        return 'done' in out.lower() or 'exit' in out.lower() or 'unkwn' in out.lower() or 'zombi' in out.lower() or\
               'is not found' in out.lower()

    async def isExist(self, jobID):
        returncode, out, err = await self.connector.execute(f'bjobs {jobID}')
        return 'run' in out.lower() or 'pend' in out.lower()

    async def kill(self, jobID):
        returncode, out, err = await self.connector.execute(f'bkill {jobID}')
        #logger.info(f'Process with jobID={jobID}  killed.')
