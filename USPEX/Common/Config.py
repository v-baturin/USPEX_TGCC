'''
@file        Config.py
@author:     Artem Samtsevich
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    samtsevichartem@gmail.com
@date        December 2017
@brief       Abstract class for engines of calculations in USPEX, like USPEX, VCNEB ...
'''

import json
from copy import copy
from abc import ABCMeta, abstractmethod


class Config(object):
    __metaclass__ = ABCMeta

    name = None

    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def isGoodSystem(self, system):
        pass

    def toJSON(self) -> str:
        return json.dumps(self.toDICT())

    def toDICT(self) -> dict:
        return copy(self.__dict__)

    @staticmethod
    def fromJSON(repr : str):
        return Config.fromDICT(json.loads(repr))

    @classmethod
    def fromDICT(cls, dct : dict):
        config = cls()
        config.__dict__ = copy(dct)
        return config