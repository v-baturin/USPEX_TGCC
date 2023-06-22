class RadialDistributionFunctions:
    def __init__(self, utility) -> None:
        self.utility = utility

    def structureFingerprint(self, system):
        """
        For using in **Fitness** infrastructure

        :param system: dictionary describing system.

        :return: calculate or retrieve structure fingerprint of a system.
        """
        if not 'radialDistribitionUtility.structureFingerprint' in system:
            self.utility.calcFingerprint(system)
        return system['radialDistribitionUtility.structureFingerprint']

    def complexFingerprint(self, system):
        """
        For using in **Fitness** infrastructure

        :param system: dictionary describing system.

        :return: calculate or retrieve structure fingerprint of a system.
        """
        if not 'radialDistribitionUtility.complexFingerprint' in system:
            self.utility.calcFingerprint(system)
        return system['radialDistribitionUtility.complexFingerprint']

    def order(self, system):
        """
        For using in **Fitness** infrastructure

        :param system: dictionary describing system.

        :return: calculate or retrieve list of atomic *local orders* of a system.
            *Local order* is a measure of atom surrounding being regular.
        """
        if not 'radialDistribitionUtility.order' in system:
            self.utility.calcFingerprint(system)
        return system['radialDistribitionUtility.order']

    def averageOrder(self, system):
        """
        For using in **Fitness** infrastructure

        :param system: dictionary describing system.

        :return: calculate or retrieve average atomic *local order* of a system.
            *Local order* is a measure of atom surrounding being regular.
        """
        if not 'radialDistribitionUtility.averageOrder' in system:
            self.utility.calcFingerprint(system)
        return system['radialDistribitionUtility.averageOrder']

    def structureOrder(self, system):
        """
        For using in **Fitness** infrastructure

        :param system: dictionary describing system.

        :return: calculate or retrieve *structure order* of a system.
            *Structure order* is a measure of structure being regular.
        """
        if not 'radialDistribitionUtility.structureOrder' in system:
            self.utility.calcFingerprint(system)
        return system['radialDistribitionUtility.structureOrder']

    def quasientropy(self, system):
        """
        For using in **Fitness** infrastructure

        :param system: dictionary describing system.

        :return: calculate quasientropy of structure.
        """

        if not 'radialDistribitionUtility.quasientropy' in system:
            self.utility.calcFingerprint(system)
        return system['radialDistribitionUtility.quasientropy']
