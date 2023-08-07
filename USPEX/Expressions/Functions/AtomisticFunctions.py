class AtomisticFunctions:

    def __init__(self, utility) -> None:
        self.utility = utility

    def environments(self, system):
        return None

    def structure(self, system):
        structure, disassembler = self.utility.atomicDisassemblerType.assemble(system)
        system.setProperty('disassembler', disassembler, extension='atomistic')
        return structure

    def disassembler(self, system):
        structure, disassembler = self.utility.atomicDisassemblerType.assemble(system)
        system.setProperty('structure', structure, extension='atomistic')
        return disassembler

    def set(self, system, prop: str, value):
        if prop == 'structure':
            return system['atomistic.disassembler'].disassemble(value)
        else:
            return {}
