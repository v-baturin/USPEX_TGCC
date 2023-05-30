class AtomisticStage:

    executorType = None

    @classmethod
    def registerTypes(cls, executorType):
        cls.executorType = executorType

    def __init__(self, tag, source=None, perturbate: bool = False, environmentStyle=None, vacuumSize=0, **kwargs):
        self.tag = tag
        self.source = source
        self.perturbate = perturbate
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
