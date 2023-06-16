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
        sink.setProperty('ID', source['ID'])
        sink.setProperty('howCome', source['howCome'])
        sink.setProperty('parent', source['parent'])
        sink.setProperty('molecules', source['molecules'])
        sink.setProperty('cell', source['cell'])
        if self.environmentStyle != 'noEnvironment' and 'environments' in source:
            sink.setProperty('environments', source['environments'])
        sink.setProperty('vacuumSize', self.vacuumSize)
        sink.setProperty('externalPressure', source['externalPressure'])
        if self.perturbate:
            sink.updateAtomicStructure(
                sink.getAtomicStructure().getPerturbatedStructure(sink['disassembler'].fixedIndices))
        await self.executor.run(sink, sink)
        self.target.constraints.systemCheckAndFix(sink)
        self.checkAndFixMolecules(source, sink)

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




