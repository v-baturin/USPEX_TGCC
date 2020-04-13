"""
USPEX.Common.Selection
======================

Abstract class for calculation engines

.. codeauthor:: Artem Samtsevich <samtsevichartem@gmail.com>
"""


class Selection(object):
    """
    Abstract class for calculation engines.
    """
    knownSelectionTypes = {}
    
    def __init__(self, target, type: str, **kwargs):
        self.config = kwargs
        self.createPopulation = self.knownSelectionTypes[type](target, **kwargs)

    @classmethod
    def registerSelection(cls, name: str, selectionType: type):
        assert name not in cls.knownSelectionTypes
        cls.knownSelectionTypes[name] = selectionType
