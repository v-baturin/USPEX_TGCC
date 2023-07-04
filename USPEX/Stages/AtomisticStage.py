import logging

import numpy as np
from ase.geometry import get_distances

logger = logging.getLogger(__name__)

class AtomisticStage:

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
        self.executor = self.executorType(tag=tag, **kwargs)

    async def run(self, source, sink):
        if self.environmentStyle != 'noEnvironment' and 'environments' in source:
            structure, disassembler = source.atomicDisassemblerType.assemble(source['molecules'], source['cell'],
                                                                             source['environments'])
        else:
            structure, disassembler = source.atomicDisassemblerType.assemble(source['molecules'], source['cell'])
        if self.perturbate:
            structure = structure.getPerturbatedStructure(disassembler.fixedIndices)
        intermediate = type(source)(extensions=source.extensions,
                                    ID=source['ID'],
                                    vacuumSize=self.vacuumSize,
                                    externalPressure=source['externalPressure'],
                                    **disassembler.disassemble(structure))

        sink.setProperty('howCome', source['howCome'])
        sink.setProperty('parent', source['parent'])
        await self.executor.run(intermediate, sink)
        self.systemCheckAndFix(sink)
        self.checkAndFixMolecules(source, sink)

    def systemCheckAndFix(self, system):
        """
        Checks if given system complies set up constraints.
        If it does, make surtain adjustments, like align the system along required axis.
        :param system: system to be checked and fixed
        """
        structure = system.getAtomicStructure()
        minDistMatrix = self.target.utilities.bondUtility.getDistances(structure.getAtomTypes(),
                                                      self.target.utilities.conditions.externalPressure)
        goodStructure = self.target.utilities.simpleMoleculeUtility.checkMinDistances(system, minDistMatrix)\
                        and self.target.utilities.cellUtility.isGoodCell(system['cell'])
        # and self.compositionSpace.isGoodComposition(self.simpleMoleculeUtility.composition(system))
        if goodStructure:
            goodStructure = goodStructure and self.target.utilities.bondUtility.isConnected(structure)
            cell = structure.getRectifiedCell()
            coordinates = cell.cartesianToFractional(structure.getCartesianCoordinates())
            if self.target.utilities.cellUtility.getDim() == 1 or self.target.utilities.cellUtility.getDim() == 2:
                cell = cell.getAlignedCell(self.target.utilities.cellUtility.getAxis())
            structure = type(structure).initFromFractionalCoordinates(structure.getAtomTypes(), coordinates, cell)
            system.updateAtomicStructure(structure)
        return goodStructure

    def checkAndFixMolecules(self, source, sink):
        correctorDict = dict()
        for i, molSink in enumerate(sink['molecules']):
            if len(molSink) > 1:
                molSource = source['molecules'][i]
                distMatSource = molSource.getAllDistances()
                cartCoordsSink  =  molSink.getCartesianCoordinates()
                distMatSink = get_distances(cartCoordsSink, cell=sink['cell'].getCellVectors(),
                              pbc=sink['cell'].getPBC())[1]
                diff = np.max(np.abs(distMatSink - distMatSource) / (distMatSource + np.eye(len(distMatSource))))
                distMatSinkNoPBC = molSink.getAllDistances()
                diffNoPBC = np.max(np.abs(distMatSinkNoPBC - distMatSource) /
                                   (distMatSource + np.eye(len(distMatSource))))
                if self.target.utilities.simpleMoleculeUtility.checkIntegrityType == 'rigid':
                    if diff > self.target.utilities.simpleMoleculeUtility.integrityTol:
                        logger.info(f'system {source["ID"]}: broken molecule detected')
                        sink.setProperty('isBad', True)
                        break
                if diffNoPBC - diff > 1e-5:
                    logger.debug(f'system {source["ID"]}: wrapped molecule detected, unwrapping')# ith molecule is wrapped
                    fractSource = source['cell'].cartesianToFractional(molSource.getCartesianCoordinates())
                    fractSink = sink['cell'].cartesianToFractional(molSink.getCartesianCoordinates())
                    wrapping = np.round(fractSink - fractSource)
                    newFractSink = fractSink - wrapping
                    correctorDict[i] = sink['cell'].fractionalToCartesian(newFractSink)
        if correctorDict:
            sink_molecules = sink['molecules']
            for i, coords in correctorDict.items():
                badMol = sink_molecules[i]
                sink_molecules[i] = type(badMol)(atomTypes=badMol.getAtomTypes(), coordinates=coords,
                                                 cell = badMol.getCell(),
                                                 zmatrixConfig = badMol.getZmatrixConfig())
            sink.setProperty('molecules', sink_molecules)




