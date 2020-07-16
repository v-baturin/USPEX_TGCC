"""
USPEX.Common.Atomistic.CrystalPool
==================================

Contains configuration of Crystal space

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import logging
logger = logging.getLogger(__name__)


from ..SystemPool import SystemPool
from .CompositionSpace import CompositionSpace


class CrystalPool(SystemPool):
    """
    This class contains configuration of Crystal space, list of systems
    already studied in the search, current result of the search.

    :cvar MAX_FORMATION_ENERGY:
        if the formation energy of the system is greater than this value,
        then the system is not added to extendedConvexHull.
    """


    def __init__(self, **kwargs):
        """
        Initializes the class.
        """
        super().__init__()

        self.compositionSpace = CompositionSpace(**kwargs)

