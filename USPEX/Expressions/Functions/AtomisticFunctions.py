class AtomisticFunctions:

    def __init__(self, utility) -> None:
        self.utility = utility

    def environments(self, system):
        return None

    def structure(self, system):
        structure, disassembler = self.utility.atomicDisassemblerType.assemble(system)
        system['atomistic.disassembler'] = disassembler
        return structure

    def disassembler(self, system):
        structure, disassembler = self.utility.atomicDisassemblerType.assemble(system)
        system['atomistic.structure'] = structure
        return disassembler

    def set(self, system, prop, value):
        if prop == 'structure':
            for key, subvalue in system['atomistic.disassembler'].disassemble(value).items():
                system[f'atomistic.{key}'] = subvalue
