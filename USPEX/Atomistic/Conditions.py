"""
USPEX.Atomistic.Conditions
==========================
"""


class Conditions:
    """
    Class describing conditions such as external pressure.
    It provides methods to estimate some system properties under provided conditions.
    """

    def __init__(self, externalPressure = 0.0001):
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
        system['externalPressure'] = self.externalPressure