from copy import copy

def usp(system, data, property, prefix):
    if prefix is None:
        if property == 'system':
            system.update(data)
        else:
            system[property] = data
    else:
        system[f'{prefix}.{property}'] = data


class AtomisticStage:

    executorType = None
    atomicDisassemblerType = None

    @classmethod
    def registerTypes(cls, executorType, atomicDisassemblerType):
        cls.executorType = executorType
        cls.atomicDisassemblerType = atomicDisassemblerType

    def __init__(self, perturbate: bool = False, environmentStyle=None, inStyle=None, vacuumSize=0, **kwargs):
        self.perturbate = perturbate
        self.environmentStyle = environmentStyle
        self.inStyle = inStyle
        self.vacuumSize = vacuumSize
        self.executor = self.executorType(**kwargs)

    async def run(self, system):
        structure, disassembler = self.atomicDisassemblerType.assemble(**system,
                                                                       style=self.environmentStyle,
                                                                       inStyle=self.inStyle,
                                                                       vacuumSize=self.vacuumSize)
        if self.perturbate:
            structure = structure.getPerturbatedStructure(disassembler.fixedIndices)
        system = copy(system)
        system['structure'] = structure
        system['disassembler'] = disassembler
        results = await self.executor.run(system)
        del system['structure']
        del system['disassembler']
        if 'structure' in results:
            usp(system, disassembler.disassemble(results.pop('structure')), 'system', self.environmentStyle)
        for key, value in results.items():
            usp(system, value, key, self.environmentStyle)
        return system
