'''
@file        TaskManager.py
@author:     Artem Samtsevich
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    samtsevichartem@gmail.com
@date        15 July 2016
@brief       Abstract class for task manager.
'''

from abc import abstractmethod

#from ..Common.Connector import Connector


class TaskManager(object):
    '''

    '''
    shortname = None

    @abstractmethod
    async def submit(self, command: str, jobname: str, input: str, output: str, error: str, calcFolder: str) -> int:
        pass

    @abstractmethod
    async def isReady(self, jobID):
        pass

    @abstractmethod
    async def isExist(self, jobID):
        pass

    @abstractmethod
    async def kill(self, jobID):
        pass
