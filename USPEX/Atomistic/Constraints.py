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
        self.bondUtiity = utilities.bondUtility
        self.conditions = utilities.conditions

    def systemCheckAndFix(self, system):
        """
        Checks if given system complies set up constraints.
        If it does, make surtain adjustments, like align the system along required axis.
        :param system: system to be checked and fixed
        """
        atomSymbols, atomDistances, disassembler = self.simpleMoleculeUtility.getMinDistances(**system)
        minDistMatrix = self.bondUtiity.getDistances(atomSymbols, self.conditions.externalPressure)
        for inds in disassembler.envIndices:
            atomDistances[tuple(np.meshgrid(inds, inds))] = minDistMatrix[tuple(np.meshgrid(inds, inds))]
        goodStructure = np.all(atomDistances >= minDistMatrix) and self.cellUtility.isGoodCell(system['cell'])
        # composition = self.simpleMoleculeUtility.composition(system)
        # and self.compositionSpace.isGoodComposition(composition)
        if goodStructure:
            structure, disassembler = self.simpleMoleculeUtility.atomicDisassemblerType.assemble(**system)
            goodStructure = goodStructure and self.bondUtiity.isConnected(structure)
            cell = structure.getRectifiedCell()
            coordinates = cell.cartesianToFractional(structure.getCartesianCoordinates())
            if (self.cellUtility.getDim() == 1 or self.cellUtility.getDim() == 2):
                cell = cell.getAlignedCell(self.cellUtility.getAxis())
            structure = type(structure).initFromFractionalCoordinates(structure.getAtomTypes(), coordinates, cell)
            system.update(disassembler.disassemble(structure))
        return goodStructure
