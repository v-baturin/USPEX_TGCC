import logging

import numpy as np
from ase.geometry import get_distances

MAX_RELATIVE_DIST_DEVIATION = 0.1

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
        self.fixMoleculesWrapping(source, sink)

    @staticmethod
    def fixMoleculesWrapping(source, sink):
        correctorDict = dict()
        for i, molSink in enumerate(sink.system['molecules']):
            if len(molSink) > 1:
                molSource = source.system['molecules'][i]
                cartCoordsSource = molSource.getCartesianCoordinates()
                cartCoordsSink  =  molSink.getCartesianCoordinates()
                distMatSource = get_distances(cartCoordsSource, cell=source.system['cell'].getCellVectors(),
                                  pbc=source.system['cell'].getPBC())[1]
                distMatSink = get_distances(cartCoordsSink, cell=sink.system['cell'].getCellVectors(),
                              pbc=sink.system['cell'].getPBC())[1]
                diff = np.max(np.abs(distMatSink - distMatSource) / (distMatSource + np.eye(len(distMatSource))))
                if diff > MAX_RELATIVE_DIST_DEVIATION:
                    logger.info(f'system {source["ID"]}: broken molecule detected')
                    sink.setProperty('isBad', True)
                    return
                distMatSourceNoPBC = get_distances(cartCoordsSource, cell=source.system['cell'].getCellVectors(),
                                                pbc=(0, 0, 0))[1]
                distMatSinkNoPBC = get_distances(cartCoordsSink, cell=sink.system['cell'].getCellVectors(),
                                              pbc=(0, 0, 0))[1]
                diffNoPBC = np.max(np.abs(distMatSinkNoPBC - distMatSourceNoPBC) /
                                   (distMatSourceNoPBC + np.eye(len(distMatSourceNoPBC))))
                if diffNoPBC - diff > 1e-5:  # ith molecule is wrapped
                    fractSource = source.system['cell'].cartesianToFractional(molSource.getCartesianCoordinates())
                    fractSink = sink.system['cell'].cartesianToFractional(molSink.getCartesianCoordinates())
                    wrapping = np.round(fractSink - fractSource)
                    newFractSink = fractSink - wrapping
                    correctorDict[i] = sink.system['cell'].fractionalToCartesian(newFractSink)
        if correctorDict:
            sink_molecules = sink.system['molecules']
            for i, coords in correctorDict.items():
                badMol = sink_molecules[i]
                sink_molecules[i] = type(badMol)(atomTypes=badMol.getAtomTypes(), coordinates=coords,
                                                 cell = badMol.getCell(),
                                                 zmatrixConfig = badMol.getZmatrixConfig())
            sink.setProperty('molecules', sink_molecules)




