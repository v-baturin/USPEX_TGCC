'''
@file        SHELL_Test.py
@author:     Pavel Bushlanov
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    paulbusj@mail.ru
@date        9 September 2016
@brief       Class for testing SHELL task manager.
'''

import unittest

import asyncio
import shutil

from pathlib import Path

from ..SHELL import SHELL
from ...Connector import Connector

TESTPATH = Path(__file__).parent
Nchan = 40


class SHELL_Test(unittest.TestCase):
    '''

    '''

    @classmethod
    def setUpClass(cls):
        #cls.command_exec = 'while true; do    echo "hello";    sleep 2; done'
        cls.command_exec = 'sleep 20'

    # def test_submit_local(self):
    #     taskManager = SHELL()
    #     #command_exec = 'while true; do    echo "hello";    sleep 2; done'
    #     coros = []
    #     for i in range(Nchan):
    #         command_exec = "echo hello{}; sleep 5".format(i)
    #         jobscript = taskManager.prepareSubmission(command_exec, 'TestJob', 'input', 'output', 'error')
    #         folder = 'folder{}/'.format(i)
    #         if os.path.exists(folder):
    #             shutil.rmtree(folder)
    #         os.mkdir(folder)
    #         with open('folder{}/input'.format(i),'wt') as f:
    #             pass
    #         coros.append(taskManager.submit(jobscript, folder))
    #     loop = asyncio.get_event_loop()
    #     jobIDs = loop.run_until_complete(asyncio.gather(*coros))
    #     for jobID in jobIDs:
    #         self.assertTrue(jobID == 0)
    #     for i in range(Nchan):
    #         with open('folder{}/output'.format(i), 'rt') as f:
    #             output = f.read()
    #         self.assertEqual(output, 'hello{}\n'.format(i))
    #         shutil.rmtree('folder{}'.format(i))

    async def coro(self, folder: Path):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir()
        input, output, error = 'input', 'output', 'error'
        with open(folder/input, 'wt') as f:
            pass
        await self.taskManager.connector.sync_l2r(folder)
        jobID = await self.taskManager.submit(self.command_exec, 'TestJob', input, output, error, folder)
        self.assertTrue(jobID == 0)
        #await self.taskManager.kill(jobID)
        #await self.taskManager.connector.copyFromRemote(output, output)
        #await self.taskManager.connector.copyFromRemote(error, error)
        await self.taskManager.connector.sync_r2l(folder)
        await self.taskManager.connector.clean(folder)
        shutil.rmtree(folder, ignore_errors=True)

    def test_submit_kill_isExist_remote(self):
        # connector = Connector(domain='192.168.88.245', username='difron', known_hosts=None,
        #                       client_keys=['{}/id_rsa'.format(TESTPATH)], remoteFolder='~/USPEX_docker_tests')

        self.taskManager = SHELL(Connector(domain = 'localhost', known_hosts=None))
        coros = []
        for i in range(Nchan):
            folder = Path(f'folder{i}')
            coros.append(self.coro(folder))
        loop = asyncio.get_event_loop()
        isExists = loop.run_until_complete(asyncio.gather(*coros))
        for i in range(Nchan):
            folder = Path(f'folder{i}')
            self.assertFalse(folder.exists())

    def test_submit_kill_isExist_local(self):
        self.taskManager = SHELL(Connector())
        coros = []
        for i in range(Nchan):
            folder = Path(f'folder{i}')
            coros.append(self.coro(folder))
        loop = asyncio.get_event_loop()
        isExists = loop.run_until_complete(asyncio.gather(*coros))
        for i in range(Nchan):
            folder = Path(f'folder{i}')
            self.assertFalse(folder.exists())


        # coros = []
        # for i in range(Nchan):
        #     command_exec = "echo hello{}; sleep 5".format(i)
        #     jobscript = self.taskManager.prepareSubmission(command_exec, 'TestJob', 'input', 'output', 'error')
        #     folder = 'folder{}/'.format(i)
        #     if os.path.exists(folder):
        #         shutil.rmtree(folder)
        #     os.mkdir(folder)
        #     with open('folder{}/input'.format(i),'wt') as f:
        #         pass
        #     coros.append(self.taskManager.submit(jobscript, folder))
        # loop = asyncio.get_event_loop()
        # jobIDs = loop.run_until_complete(asyncio.gather(*coros))
        # for jobID in jobIDs:
        #     self.assertTrue(jobID == 0)
        # for i in range(Nchan):
        #     with open('folder{}/output'.format(i), 'rt') as f:
        #         output = f.read()
        #     self.assertEqual(output, 'hello{}\n'.format(i))
        #     shutil.rmtree('folder{}'.format(i))
