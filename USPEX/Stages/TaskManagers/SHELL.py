"""
USPEX.Stages.TaskManagers.SHELL
===============================

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>

"""
import logging
import os

logger = logging.getLogger(__name__)


class SHELL:


    shortname = 'SHELL'

    def __init__(self, connector):
        """
        :param connector: for remote submission
        :return:
        """

        self.connector = connector

    async def submit(self, command: str, jobname: str, input: str, output: str, error: str, calcFolder: str) -> int:

        # if not os
        with open(os.path.join(calcFolder, input), 'r') as fi,\
                open(os.path.join(calcFolder, output), 'w') as fo,\
                open(os.path.join(calcFolder, error), 'w') as fe:
            returncode, out, err= await self.connector.execute(command, stdin=fi, stdout=fo, stderr=fe, cwd=calcFolder)
            logger.debug('process returned code {}'.format(returncode))
            if returncode != 0:
                logger.error(err)
                logger.error(out)

        return 0 if returncode == 0 else -1

    async def isReady(self, jobID : int):
        return True

    async def isExist(self, jobID : int):
        return False

    async def kill(self, jobID : int):
        pass
