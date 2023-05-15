"""
USPEX.Stages.TaskManagers.Screen_TM
===================================

.. codeauthor:: Artem Samtsevich <samtsevichartem@gmail.com>

"""
import logging

from pathlib import Path

from .Screen import Screen, list_screens_id, kill_screen_by_id

logger = logging.getLogger(__name__)


class Screen_TM:
    type = 'Screen'

    def __init__(self, connector=None):
        self.connector = connector

    # def _prepareSubmission(self, COMMAND_EXEC : str, JOB_NAME : str,
    #                              inputFile : str, outputFile : str, errorFile : str) -> str:
    #     '''
    #     Preparing jobscript for submission
    #     :param commandExec:
    #     :param jobName:
    #     :return: jobscript as string
    #     '''
    #
    #     content = ''
    #     for line in self.header.split('\n'):
    #         if ' -j ' in line.lower():
    #             logging.info('Job name found in HEADER will be overwritten')
    #         elif ' -i ' in line.lower():
    #             logging.info('Input file name found in HEADER will be overwritten')
    #         elif ' -o ' in line.lower():
    #             logging.info('Output file name found in HEADER will be overwritten')
    #         elif ' -e ' in line.lower():
    #             logging.info('Error file name found in HEADER will be overwritten')
    #         else:
    #             content += line + '\n'
    #     content += '#SBATCH -J  {}\n'.format(JOB_NAME)
    #     content += '#SBATCH -i  {}\n'.format(inputFile)
    #     content += '#SBATCH -o  {}\n'.format(outputFile)
    #     content += '#SBATCH -e  {}\n'.format(errorFile)
    #     content += '\n' + COMMAND_EXEC + '\n\n'
    #
    #     return ''.join(content)

    async def submit(self, command: str, jobname: str, input: str, output: str, error: str, calcFolder: Path) -> int:
        # jobscript = self._prepareSubmission(command, jobname, input, output, error)
        # with open(os.path.join(calcFolder, 'jobscript'), 'wt') as f:
        #     f.write(jobscript)
        # await self.connector.sync_l2r(os.path.join(calcFolder, 'jobscript'))

        screen = Screen(name=jobname, connector=self.connector)
        ID = await screen.run_and_exit(command, cwd=calcFolder)
        # await screen.initialize(cwd=calcFolder)
        # await screen.send_commands(command, cwd=calcFolder)
        return ID

    def _parseJobID(self, output : str, error: str) -> int:
        """

        :param output:
        :param error:
        :return: jobID
        """
        return int(output.split()[-1])

    async def isReady(self, jobID : int):
        return jobID not in await list_screens_id(self.connector)

    async def isExist(self, jobID : int):
        return jobID in await list_screens_id(self.connector)

    async def kill(self, jobID : int):
        # screens_with_id = [screen for screen in await list_screens(connector=self.connector, initialize=False) if await screen.id == jobID]
        # if not screens_with_id:
        #     return True
        # else:
        #     await screens_with_id[0].kill()

        await kill_screen_by_id(connector=self.connector, id=jobID)

        #logging.info('Process with jobID={}  killed.'.format(jobID))
