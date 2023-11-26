import logging

import numpy as np
from itertools import product
from ase.geometry import get_distances

from ..Optimizers.PoolEntry import PoolEntry

logger = logging.getLogger(__name__)


class AtomisticStage:
    EV_PER_CUBIC_ANGSTREM_PER_GPA = 1 / 160.21766208

    executorType = None

    @classmethod
    def registerTypes(cls, executorType):
        cls.executorType = executorType

    def __init__(self, tag, source=None, perturbate: bool = False, target=None, environmentStyle=None, vacuumSize=0,
                 targetProperties=None, **kwargs):
        self.tag = tag
        self.source = source
        self.perturbate = perturbate
        self.target = target
        self.environmentStyle = environmentStyle
        self.vacuumSize = vacuumSize
        self.targetProperties = targetProperties if targetProperties is not None else ['structure', 'enthalpy']
        self.kwargs = kwargs
        self.executor = self.executorType(tag=tag, targetProperties=self.targetProperties, **kwargs)

    async def run(self, system: PoolEntry):
        if self.environmentStyle != 'noEnvironment':
            structure = system.getProperty('structure', extension='atomistic', suffix=self.source)
            disassembler = system.getProperty('disassembler', extension='atomistic', suffix=self.source)
        else:
            source = system.getFlavour(self.source)
            structure, disassembler = source.extensions['atomistic'].atomicDisassemblerType.assemble({
                'atomistic.molecules': source[f'atomistic.molecules'],
                'atomistic.cell': source[f'atomistic.cell']
            })
        if self.perturbate:
            structure = structure.getPerturbatedStructure(disassembler.fixedIndices)
        intermediate = disassembler.disassemble(structure)
        intermediate['.ID'] = system.ID
        intermediate['.vacuumSize'] = self.vacuumSize
        intermediate['.externalPressure'] = system.getProperty('externalPressure', suffix='origin')
        intermediate = system.flavourFactory(**intermediate)

        try:
            result = await self.executor.run(system.ID, intermediate)
        except Exception as ex:
            logger.warning(f'system {system.ID} error in relaxation:')
            logger.exception(ex)
            system.setProperty('isBad', True, suffix=self.tag)
            return
        disassembler = intermediate.getProperty('disassembler', extension='atomistic')
        result.setProperty('disassembler', disassembler, extension='atomistic')
        self.systemCheckAndFix(result)
        if 'enthalpy' in self.targetProperties and '.enthalpy' not in result:
            structure = result.getProperty('structure', extension='atomistic')
            pressure = system.getProperty('externalPressure', suffix='origin')
            energy = result.getProperty('energy')
            enthalpy = energy + structure.getCell().getVolume() * pressure * self.EV_PER_CUBIC_ANGSTREM_PER_GPA
            result.setProperty('enthalpy', enthalpy)
        system.addFlavour(self.tag, result)
        self.checkAndFixMolecules(system)

    def systemCheckAndFix(self, system):
        """
        Checks if given system complies with set up constraints.
        If it does, make certain adjustments, like align the system along required axis.
        :param system: system to be checked and fixed
        """
        structure = system.getProperty('structure', extension='atomistic')
        cell = system.getProperty('cell', extension='atomistic')
        minDistMatrix = self.target.utilities.bondUtility.getDistances(structure.getAtomTypes(),
                                                                       self.target.utilities.conditions.externalPressure)
        goodStructure = self.target.utilities.simpleMoleculeUtility.checkMinDistances(system, minDistMatrix) \
                        and self.target.utilities.cellUtility.isGoodCell(cell)
        # and self.compositionSpace.isGoodComposition(self.simpleMoleculeUtility.composition(system))
        if goodStructure:
            goodStructure = goodStructure and self.target.utilities.bondUtility.isConnected(structure)
            cell = structure.getRectifiedCell()
            coordinates = cell.cartesianToFractional(structure.getCartesianCoordinates())
            if self.target.utilities.cellUtility.getDim() == 1 or self.target.utilities.cellUtility.getDim() == 2:
                cell = cell.getAlignedCell(self.target.utilities.cellUtility.getAxis())
            structure = type(structure).initFromFractionalCoordinates(structure.getAtomTypes(), coordinates, cell,
                                                                      edges=structure.edges)
            system.setProperty('structure', structure, extension='atomistic')
        system.setProperty('isBad', not goodStructure)

    def checkAndFixMolecules(self, system):
        correctorDict = dict()
        moleculesSink = system.getProperty('molecules', extension='atomistic', suffix=self.tag)
        moleculesSource = system.getProperty('molecules', extension='atomistic', suffix=self.source)
        cellSink = system.getProperty('cell', extension='atomistic', suffix=self.tag)
        cellSource = system.getProperty('cell', extension='atomistic', suffix=self.source)
        for i, molSink in enumerate(moleculesSink):
            nAtoms = len(molSink)
            if nAtoms > 1:
                molSource = moleculesSource[i]
                if self.target.utilities.simpleMoleculeUtility.checkIntegrityType == 'rigid':
                    ADJ_MAT = np.ones((nAtoms, nAtoms)) - np.eye(nAtoms)  # TODO: nontrivial ADJ_MAT for flexible mols
                distMatSource = molSource.getAllDistances() * ADJ_MAT
                checkedAndFixedGen = self.checkAndFixWrap(cellSource, molSource, distMatSource, cellSink, molSink, ADJ_MAT)
                for k_try, newCoords in enumerate(checkedAndFixedGen):
                    if np.all(np.abs(get_distances(newCoords)[1] * ADJ_MAT - distMatSource) /
                              (distMatSource + np.eye(nAtoms)) <
                              self.target.utilities.simpleMoleculeUtility.integrityTol):
                        if k_try > 0:
                            correctorDict[i] = newCoords
                        break
                else:
                    logger.info(f'system {system["ID"]}: broken molecule detected')
                    system.setProperty('isBad', True, suffix=self.tag)
                    break

        for i, coords in correctorDict.items():
            badMol = moleculesSink[i]
            moleculesSink[i] = type(badMol)(atomTypes=badMol.getAtomTypes(), coordinates=coords,
                                            cell=badMol.getCell(),
                                            edges=badMol.edges)
        if correctorDict:
            logger.debug(
                f'system {system["ID"]}: unwrapped {len(correctorDict)} molecules')
            system.setProperty('molecules', moleculesSink, extension='atomistic', suffix=self.tag)

    def checkAndFixWrap(self, cellSource, molSource, distMatSource, cellSink, molSink, adjMatrix):
        newMolSinkCoords = molSink.getCartesianCoordinates()
        yield newMolSinkCoords
        fractSource = cellSource.cartesianToFractional(molSource.getCartesianCoordinates())
        fractSink = cellSink.cartesianToFractional(newMolSinkCoords)
        wrapping = np.round(fractSink - fractSource)
        newFractSink = fractSink - wrapping
        yield cellSink.fractionalToCartesian(newFractSink)  # first guess: dewrap if xfrac changes more than by 0.5
        nAtoms = len(molSource)
        distMatSink = molSink.getAllDistances() * adjMatrix
        relativeDiff = np.abs(distMatSink - distMatSource) / (distMatSource + np.eye(nAtoms))
        isBadDist = relativeDiff >= self.target.utilities.simpleMoleculeUtility.integrityTol
        for idxBadDistA in range(len(molSource)):
            for idxBadDistB in np.where(isBadDist[idxBadDistA, idxBadDistA:])[0]:
                coordA = newMolSinkCoords[idxBadDistA]
                coordB = newMolSinkCoords[idxBadDistB]
                sourceDist = distMatSource[idxBadDistA, idxBadDistB]
                newCoordsB = self.dewrapAtomB(coordA, coordB, sourceDist, cellSink)
                if newCoordsB is not None:
                    newMolSinkCoords[idxBadDistB] = newCoordsB
                    isBadDist[idxBadDistB] = False
                    isBadDist[:, idxBadDistB] = False
                else:
                    return
        yield newMolSinkCoords

    def dewrapAtomB(self, coordA, coordB, sourceDist, cellSink):
        allwrappings = np.array(list(product(*[range(i, f + 1) for i, f in zip((-1, -1, -1), (1, 1, 1))])))
        wrappingsOfB = coordB - allwrappings @ cellSink.getCellVectors()
        good_wrap_idx = np.where(np.abs(get_distances(wrappingsOfB, coordA)[1] - sourceDist) / sourceDist <
                                 self.target.utilities.simpleMoleculeUtility.integrityTol)[0]
        if len(good_wrap_idx) == 1:
            return wrappingsOfB[good_wrap_idx[0]]
        elif len(good_wrap_idx) > 1:
            logger.warning("atom with frac coords {:.3f} {:.3f} {:.3f}: ambiguous dewrapping".format(*cellSink.cartesianToFractional(coordB)))
