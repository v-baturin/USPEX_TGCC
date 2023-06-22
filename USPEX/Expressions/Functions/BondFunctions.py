class BondFunctions:
    def __init__(self, utility) -> None:
        self.utility = utility

    def hardness(self, system):
        structure, disassembler = self.utility.disassemblerType.assembe(**system)
        bonds = self.utility.getMinimalGraphBonds(structure)
        return self.utility.calcHardness(structure, bonds)

