import numpy as np


class Constraints:
    def __init__(self, utilities):
        self.cellUtility = utilities.cellUtility
        self.compositionSpace = utilities.compositionSpace
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.ionDistances = utilities.ionDistances
        self.conditions = utilities.conditions

    def systemCheckAndFix(self, system):
        cell = system['cell']
        molecules = system['molecules']
        atomSymbols, atomDistances = self.simpleMoleculeUtility.getMinDistances(molecules, cell)
        minDistMatrix = self.ionDistances.getDistances(atomSymbols, self.conditions.externalPressure)
        composition = self.simpleMoleculeUtility.composition(system)
        goodStructure = np.all(atomDistances >= minDistMatrix) \
                        and self.compositionSpace.isGoodComposition(composition) # and self.cellUtility.isGoodCell(cell)
        if goodStructure and (self.cellUtility.getDim() == 1 or self.cellUtility.getDim() == 2):
            structure, disassembler = self.simpleMoleculeUtility.structureType.assemble(**system)
            cell = structure.getRectifiedCell()
            coordinates = cell.cartesianToFractional(structure.getCartesianCoordinates())
            cell = cell.getAlignedCell(self.cellUtility.getAxis())
            structure = type(structure).initFromFractionalCoordinates(structure.getAtomTypes(), coordinates, cell)
            system.update(disassembler.disassemble(structure))
        return goodStructure
