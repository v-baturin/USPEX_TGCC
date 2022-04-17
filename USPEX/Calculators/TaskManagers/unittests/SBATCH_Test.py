'''
@file        SHELL_Test.py
@author:     Pavel Bushlanov
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    paulbusj@mail.ru
@date        9 September 2016
@brief       Class for testing SHELL task manager.
'''

import unittest
from ..SBATCH import SBATCH
from USPEX.Calculators.Connector import Connector

import os
import shutil
import asyncio

TESTPATH = os.path.dirname(os.path.abspath(__file__))
Nchan = 40

HEADER =  '''#!/bin/sh
#SBATCH -p normal
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH -n 1

'''


class SBATCH_Test(unittest.TestCase):
    '''

    '''



    @classmethod
    def setUpClass(cls):
        cls.command_exec = 'sleep 20'


    async def coro(self, i):
        folder = 'folder{}/'.format(i)
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.mkdir(folder)
        input = 'folder{}/input'.format(i)
        output = 'folder{}/output'.format(i)
        error = 'folder{}/error'.format(i)
        with open(input, 'wt') as f:
            pass
        await self.taskManager.connector.sync_l2r(folder)
        jobID = await self.taskManager.submit(self.command_exec, 'TestJob', 'input', 'output', 'error', folder)
        self.assertTrue(jobID > 0)
        self.assertTrue(await self.taskManager.isExist(jobID))
        await self.taskManager.kill(jobID)
        await self.taskManager.connector.sync_r2l(folder)
        await self.taskManager.connector.clean(folder)
        shutil.rmtree('folder{}'.format(i))

    def test_submit_kill_isExist_remote(self):
        self.taskManager = SBATCH(HEADER, Connector(domain = 'localhost', known_hosts=None))
        coros = []
        for i in range(Nchan):
            coros.append(self.coro(i))
        loop = asyncio.get_event_loop()
        isExists = loop.run_until_complete(asyncio.gather(*coros))

    def test_submit_kill_isExist_local(self):
        self.taskManager = SBATCH(HEADER, Connector())
        coros = []
        for i in range(Nchan):
            coros.append(self.coro(i))
        loop = asyncio.get_event_loop()
        isExists = loop.run_until_complete(asyncio.gather(*coros))
