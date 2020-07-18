"""
USPEX.Common.NoRepresentation
=============================

Empty output representation

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""


class NoRepresentation(object):
    """
    Empty output representation.
    """
    def __init__(self, **kwargs):
        pass

    def presentSystems(self, systems, optimizer):
        pass

    def presentOutput(self, populations, optimizer):
        pass

    def presentInfo(self, info):
        pass

    def presentOptimizer(self, optimizers):
        pass
