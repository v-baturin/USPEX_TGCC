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
        self.bondUtility = utilities.bondUtility
        self.conditions = utilities.conditions

    def systemCheckAndFix(self, system):
        """
        Checks if given system complies set up constraints.
        If it does, make surtain adjustments, like align the system along required axis.
        :param system: system to be checked and fixed
        """
        structure = system.getAtomicStructure()
        minDistMatrix = self.bondUtility.getDistances(structure.getAtomTypes(),
                                                      self.conditions.externalPressure)
        goodStructure = self.simpleMoleculeUtility.checkMinDistances(system, minDistMatrix)\
                        and self.cellUtility.isGoodCell(system['cell'])
        # and self.compositionSpace.isGoodComposition(self.simpleMoleculeUtility.composition(system))
        if goodStructure:
            goodStructure = goodStructure and self.bondUtility.isConnected(structure)
            cell = structure.getRectifiedCell()
            coordinates = cell.cartesianToFractional(structure.getCartesianCoordinates())
            if self.cellUtility.getDim() == 1 or self.cellUtility.getDim() == 2:
                cell = cell.getAlignedCell(self.cellUtility.getAxis())
            structure = type(structure).initFromFractionalCoordinates(structure.getAtomTypes(), coordinates, cell)
            system.updateAtomicStructure(structure)
        return goodStructure
