'''
@file        QSUB_Test.py
@author:     Michele Galasso
@copyright:  2018 Oganov's Lab. All rights reserved.
@contact:    m.galasso@yandex.com
@date        11 April 2018
@brief       Class for testing QSUB.
'''

import unittest
from ..QSUB import QSUB
from ...Connector import Connector

import os
import shutil
import asyncio

Nchan = 40

TESTPATH = os.path.dirname(os.path.abspath(__file__))


HEADER = '''#!/bin/sh
#PBS -l nodes=1:ppn=8
#PBS -l walltime=48:00:00

'''


class QSUB_Test(unittest.TestCase):
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

        self.taskManager = QSUB(HEADER, Connector(domain = 'localhost', known_hosts=None))
        coros = []
        for i in range(Nchan):
            coros.append(self.coro(i))
        loop = asyncio.get_event_loop()
        isExists = loop.run_until_complete(asyncio.gather(*coros))

    def test_submit_kill_isExist_local(self):
        self.taskManager = QSUB(HEADER, Connector())
        coros = []
        for i in range(Nchan):
            coros.append(self.coro(i))
        loop = asyncio.get_event_loop()
        isExists = loop.run_until_complete(asyncio.gather(*coros))
