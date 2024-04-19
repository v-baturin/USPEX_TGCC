from ...DataModel.Flavour import Flavour


class BondFunctions:
    def __init__(self, utility) -> None:
        self.utility = utility

    def hardness(self, system: Flavour):
        structure = system.getProperty('structure', extension='atomistic')
        bonds = self.utility.getMinimalGraphBonds(structure)
        return self.utility.calcHardness(structure, bonds)

