'''
@file        System.py
@author:     Pavel Bushlanov
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    paulbush@mail.ru
@date        21 July 2016
@brief       Base class for different systems
'''

from copy import copy
import json


class System(object):
    '''

    '''

    _isBad = False

    def markBad(self):
        self._isBad = True

    @property
    def isBad(self):
        return self._isBad


    def toJSON(self) -> str:
        '''
        Method which creates JSON representation of the structure.

        :return: String with JSON representation of the structure.
        '''
        return json.dumps(self.toDICT())

    def toDICT(self) -> dict:
        '''
        Method which creates dictionary representation of the structure.

        :return: Dictionary representing the structure.
        '''
        return copy(self.__dict__)

    @classmethod
    def fromJSON(cls, repr : str):
        '''
        Method which reconstructs AtoimicStructure from JSON representation.

        :param repr: String with JSON representation of the structure.
        '''
        return cls.fromDICT(json.loads(repr))

    @classmethod
    def fromDICT(cls, dct : dict):
        '''
        Method which reconstructs AtoimicStructure from dictionary representation.

        :param dct: Dictionary representing the structure.
        '''
        newStructure = cls()
        newStructure.__dict__.update(copy(dct))
        return newStructure
