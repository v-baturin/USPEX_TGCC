from ...DataModel.Flavour import Flavour


class AtomisticFunctions:

    def __init__(self, utility) -> None:
        self.utility = utility

    @staticmethod
    def _setDisassembled(system: Flavour):
        structure = system.getProperty('structure', extension='atomistic')
        disassembler = system.getProperty('disassembler', extension='atomistic')
        for key, value in disassembler.disassemble(structure).items():
            extension, prop, *other = key.split('.')
            assert len(other) == 0, f'Too complex property name {key}.'
            system.setProperty(prop, value, extension=extension)

    def cell(self, system: Flavour):
        self._setDisassembled(system)
        return system.getProperty('cell', extension='atomistic')

    def molecules(self, system: Flavour):
        self._setDisassembled(system)
        return system.getProperty('molecules', extension='atomistic')

    def environments(self, system: Flavour):
        if 'atomistic.cell' not in system and 'atomistic.molecules' not in system:
            self._setDisassembled(system)
            return system.getProperty('environments', extension='atomistic')
        else:
            return []

    def structure(self, system: Flavour):
        structure, disassembler = self.utility.atomicDisassemblerType.assemble(system)
        system.setProperty('disassembler', disassembler, extension='atomistic')
        return structure

    def disassembler(self, system: Flavour):
        structure, disassembler = self.utility.atomicDisassemblerType.assemble(system)
        system.setProperty('structure', structure, extension='atomistic')
        return disassembler
