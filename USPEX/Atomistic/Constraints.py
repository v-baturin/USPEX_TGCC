"""
USPEX.Atomistic.Constraints
===========================
"""

import numpy as np


class Constraints:
    """
    Class for placing constraints for structures.
    """

    def __init__(self, utilities):
        self.cellUtility = utilities.cellUtility
        self.compositionSpace = utilities.compositionSpace
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.ionDistances = utilities.ionDistances
        self.bonds = utilities.bonds
        self.conditions = utilities.conditions

    def systemCheckAndFix(self, system):
        """
        Checks if given system complies set up constraints.
        If it does, make surtain adjustments, like align the system along required axis.
        :param system: system to be checked and fixed
        """
        cell = system['cell']
        molecules = system['molecules']
        atomSymbols, atomDistances = self.simpleMoleculeUtility.getMinDistances(molecules, cell)
        minDistMatrix = self.ionDistances.getDistances(atomSymbols, self.conditions.externalPressure)
        composition = self.simpleMoleculeUtility.composition(system)
        goodStructure = np.all(atomDistances >= minDistMatrix) \
                        and self.compositionSpace.isGoodComposition(composition) # and self.cellUtility.isGoodCell(cell)
        if goodStructure:
            structure, disassembler = self.simpleMoleculeUtility.structureType.assemble(**system)
            goodStructure = goodStructure and self.bonds.isConnected(structure)
            cell = structure.getRectifiedCell()
            coordinates = cell.cartesianToFractional(structure.getCartesianCoordinates())
            if (self.cellUtility.getDim() == 1 or self.cellUtility.getDim() == 2):
                cell = cell.getAlignedCell(self.cellUtility.getAxis())
            structure = type(structure).initFromFractionalCoordinates(structure.getAtomTypes(), coordinates, cell)
            system.update(disassembler.disassemble(structure))
        return goodStructure
