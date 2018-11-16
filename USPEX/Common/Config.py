'''
@file        Config.py
@author:     Artem Samtsevich
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    samtsevichartem@gmail.com
@date        December 2017
@brief       Abstract class for engines of calculations in USPEX, like USPEX, VCNEB ...
'''

from abc import ABCMeta, abstractmethod


class Config(object):
    __metaclass__ = ABCMeta

    name = None

    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def isGoodSystem(self, system):
        pass
