import spglib


class CellFunctions:
    def __init__(self, utility) -> None:
        self.utility = utility

    @staticmethod
    def volume(system):
        """
        For using in **Fitness** infrastructure

        :param system: dictionary describing system.

        :return: calculated volume of system.
        """
        return system['cell'].getVolume()

    @staticmethod
    def area(system):
        """
        For using in **Fitness** infrastructure

        :param system: dictionary describing system.

        :return: calculated area of system.
        """
        return system['cell'].getArea()
    
    @staticmethod
    def length(system):
        """
        For using in **Fitness** infrastructure

        :param system: dictionary describing system.

        :return: calculated length of system.
        """
        return system['cell'].getLength()

    def symmetry(self, system):
        """
        For using in **Fitness** infrastructure

        :param system: dictionary describing system.

        :return: calculated symmetry of system.
        """
        structure = system.getAtomicStructure()
        cell = structure.getCell()
        lattice = cell.getCellVectors()
        coordinates = structure.getFractionalCoordinates()
        numbers = [el.z for el in structure.getAtomTypes()]
        spacegroup = spglib.get_spacegroup((lattice, coordinates, numbers), symprec=self.utility.symTolerance)
        if cell.dim == 3 and spacegroup is not None:
            symmetry = '{:7s} {:4s}'.format(*[str(x) for x in spacegroup.split()])
        else:
            symmetry = None
        return symmetry

