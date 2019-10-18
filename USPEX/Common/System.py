'''
@file        System.py
@author:     Pavel Bushlanov
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    paulbush@mail.ru
@date        21 July 2016
@brief       Base class for different systems
'''

from copy import copy


class System(object):
    '''

    '''

    _newID = 0
    _isBad = False


    def __init__(self, **kwargs):
        super(System, self).__init__(**kwargs)
        self.ID = self.getNewID()
        self.setNewID(self.ID + 1)
        self.howCome = None

    @classmethod
    def getNewID(cls):
        return cls._newID

    @classmethod
    def setNewID(cls, ID):
        cls._newID = ID

    def __hash__(self):
        return hash(self.ID)

    def markBad(self):
        self._isBad = True

    def isBad(self):
        return self._isBad

    ############################################
    # Code responds for serialization
    def __getstate__(self):
        state = copy(self.__dict__)
        state['newID'] = self.getNewID()
        return state

    def __setstate__(self, state):
        newID = state['newID']
        del state['newID']
        self.__dict__ = copy(state)
        self.setNewID(newID)
