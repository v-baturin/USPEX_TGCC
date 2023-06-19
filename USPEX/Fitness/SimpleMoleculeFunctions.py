import numpy as np
from collections import Counter


DENSITY_CONST = 1.660539


class SimpleMoleculeFunctions:

    def __init__(self, utility) -> None:
        self.utility = utility

    def moleculeTypes(self, system):
        """
        For using in **Fitness** infrastructure

        :param system: dictionary describing system.

        :return: calculated or retrieve list of types of molecules of a system.
        """
        return [self.utility.determineMoleculeType(molecule) for molecule in system['molecules']]

    def composition(self, system):
        """
        For using in **Fitness** infrastructure

        :param system: dictionary describing system.

        :return: calculated or retrieve molecular composition of a system.
        """
        return Counter(dict(zip(*np.unique(self.moleculeTypes(system), return_counts=True))))

    def density(self, system):
        cell = system['cell']
        if cell.dim == 3:
            mass = sum(e.mass*v for e, v in self.utility.getElementalComposition(self.composition(system)).items())
            return mass/cell.getVolume()*DENSITY_CONST
        else:
            return None
