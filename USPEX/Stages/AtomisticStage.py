import logging
import numpy as np
from ase.geometry import get_distances

from ..Optimizers.PoolEntry import PoolEntry

logger = logging.getLogger(__name__)


class AtomisticStage:

    EV_PER_CUBIC_ANGSTREM_PER_GPA = 1 / 160.21766208

    executorType = None

    @classmethod
    def registerTypes(cls, executorType):
        cls.executorType = executorType

    def __init__(self, tag, source=None, perturbate: bool = False, target=None, environmentStyle=None, vacuumSize=0, **kwargs):
        self.tag = tag
        self.source = source
        self.perturbate = perturbate
        self.target = target
        self.environmentStyle = environmentStyle
        self.vacuumSize = vacuumSize
        self.kwargs = kwargs
        self.executor = self.executorType(tag=tag, **kwargs)

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
        intermediate['.vacuumSize'] = self.vacuumSize
        intermediate['.externalPressure'] = system.getProperty('externalPressure', suffix='origin')
        intermediate['atomistic.disassembler'] = disassembler
        intermediate = system.flavourFactory(**intermediate)

        result = await self.executor.run(system.ID, intermediate)
        result.setProperty('disassembler', disassembler, extension='atomistic')
        self.systemCheckAndFix(result)
        if 'targetProperties' in self.kwargs and 'enthalpy' in self.kwargs['targetProperties'] \
                and '.enthalpy' not in system.getFlavour(self.tag):
            structure = result.getProperty('structure', extension='atomistic')
            pressure = system.getProperty('externalPressure', suffix='origin')
            energy = result.getProperty('energy')
            enthalpy = energy + structure.getVolume() * pressure * self.EV_PER_CUBIC_ANGSTREM_PER_GPA
            result.setProperty('enthalpy', enthalpy)
        system.addFlavour(self.tag, result)
        self.checkAndFixMolecules(system)



    def systemCheckAndFix(self, system):
        """
        Checks if given system complies set up constraints.
        If it does, make surtain adjustments, like align the system along required axis.
        :param system: system to be checked and fixed
        """
        structure = system.getProperty('structure', extension='atomistic')
        cell = system.getProperty('cell', extension='atomistic')
        minDistMatrix = self.target.utilities.bondUtility.getDistances(structure.getAtomTypes(),
                                                      self.target.utilities.conditions.externalPressure)
        goodStructure = self.target.utilities.simpleMoleculeUtility.checkMinDistances(system, minDistMatrix)\
                        and self.target.utilities.cellUtility.isGoodCell(cell)
        # and self.compositionSpace.isGoodComposition(self.simpleMoleculeUtility.composition(system))
        if goodStructure:
            goodStructure = goodStructure and self.target.utilities.bondUtility.isConnected(structure)
            cell = structure.getRectifiedCell()
            coordinates = cell.cartesianToFractional(structure.getCartesianCoordinates())
            if self.target.utilities.cellUtility.getDim() == 1 or self.target.utilities.cellUtility.getDim() == 2:
                cell = cell.getAlignedCell(self.target.utilities.cellUtility.getAxis())
            structure = type(structure).initFromFractionalCoordinates(structure.getAtomTypes(), coordinates, cell)
            system.setProperty('structure', structure, extension='atomistic')
        system.setProperty('isBad', not goodStructure)

    def checkAndFixMolecules(self, system):
        correctorDict = dict()
        moleculesSink = system.getProperty('molecules', extension='atomistic', suffix=self.tag)
        moleculesSource = system.getProperty('molecules', extension='atomistic', suffix=self.source)
        cellSink = system.getProperty('cell', extension='atomistic', suffix=self.tag)
        cellSource = system.getProperty('cell', extension='atomistic', suffix=self.source)
        for i, molSink in enumerate(moleculesSink):
            if len(molSink) > 1:
                molSource = moleculesSource[i]
                distMatSource = molSource.getAllDistances()
                cartCoordsSink  =  molSink.getCartesianCoordinates()
                distMatSink = get_distances(cartCoordsSink, cell=cellSink.getCellVectors(),
                              pbc=cellSink.getPBC())[1]
                diff = np.max(np.abs(distMatSink - distMatSource) / (distMatSource + np.eye(len(distMatSource))))
                distMatSinkNoPBC = molSink.getAllDistances()
                diffNoPBC = np.max(np.abs(distMatSinkNoPBC - distMatSource) /
                                   (distMatSource + np.eye(len(distMatSource))))
                if self.target.utilities.simpleMoleculeUtility.checkIntegrityType == 'rigid':
                    if diff > self.target.utilities.simpleMoleculeUtility.integrityTol:
                        logger.info(f'system {system["ID"]}: broken molecule detected')
                        system.setProperty('isBad', True, suffix=self.tag)
                        break
                if diffNoPBC - diff > 1e-5:
                    logger.debug(f'system {system["ID"]}: wrapped molecule detected, unwrapping')# ith molecule is wrapped
                    fractSource = cellSource.cartesianToFractional(molSource.getCartesianCoordinates())
                    fractSink = cellSink.cartesianToFractional(molSink.getCartesianCoordinates())
                    wrapping = np.round(fractSink - fractSource)
                    newFractSink = fractSink - wrapping
                    correctorDict[i] = cellSink.fractionalToCartesian(newFractSink)
        for i, coords in correctorDict.items():
            badMol = moleculesSink[i]
            moleculesSink[i] = type(badMol)(atomTypes=badMol.getAtomTypes(), coordinates=coords,
                                             cell = badMol.getCell(),
                                             zmatrixConfig = badMol.getZmatrixConfig())
        if correctorDict:
            system.setProperty('molecules', moleculesSink, extension='atomistic', suffix=self.tag)




