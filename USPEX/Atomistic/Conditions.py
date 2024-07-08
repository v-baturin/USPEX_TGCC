"""
USPEX.Atomistic.Conditions
==========================
"""

from ..Semantics.Atomistic.Conditions import Conditions as CondeitionsSemantics

class Conditions(CondeitionsSemantics):
    """
    Class describing conditions such as external pressure.
    """

    def __init__(self, externalPressure = 0.0):
        """

        :type externalPressure: float
        :param externalPressure: target pressure.
        """

        self.externalPressure = externalPressure

    def putConditions(self, system):
        """
        Put parameters into system dictionary.

        :param system: dictionary to put parameters into.

        """
        system.setProperty('externalPressure', self.externalPressure)
