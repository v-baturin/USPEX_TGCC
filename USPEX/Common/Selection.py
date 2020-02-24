'''
@file        Selection.py.py
@author:     Artem Samtsevich
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    samtsevichartem@gmail.com
@date        December 2017
@brief       Abstract class for engines of calculations in USPEX, like USPEX, VCNEB ...
'''


class Selection(object):

    knownSelectionTypes = {}
    
    def __init__(self, target, type : str, **kwargs):
        self.config = kwargs
        self.createPopulation = self.knownSelectionTypes[type](target, **kwargs)

    @classmethod
    def registerSelection(cls, name : str, selectionType : type):
        assert name not in cls.knownSelectionTypes
        cls.knownSelectionTypes[name] = selectionType