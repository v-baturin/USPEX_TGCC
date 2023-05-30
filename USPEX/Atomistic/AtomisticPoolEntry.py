class AtomisticPoolEntry:

    atomicDisassemblerType = None

    @classmethod
    def registerTypes(cls, atomicDisassemblerType):
        cls.atomicDisassemblerType = atomicDisassemblerType

    def __init__(self, **system):
        self.system = system

    def getAtomicStructure(self, prefix=None):
        system = self._getPrefixedValue(prefix)
        if 'structure' not in system:
            structure, disassembler = self.atomicDisassemblerType.assemble(**system)
            system['structure'] = structure
            system['disassembler'] = disassembler
        return system['structure']

    def updateAtomicStructure(self, structure, prefix=None):
        system = self._getPrefixedValue(prefix)
        disassembler = system.pop('disassembler')
        del system['structure']
        data = disassembler.disassemble(structure)
        system.update(data)

    def setAtomicStructure(self, structure, disassembler, prefix=None):
        self._setPrefixedValue(prefix, disassembler.disassemble(structure))

    def getProperty(self, prop, prefix=None):
        return self._getPrefixedValue(prefix, prop)

    def setProperty(self, prop, value, prefix=None):
        self._setPrefixedValue(prefix, value, prop)

    def popProperty(self, prop, prefix=None):
        value = self._getPrefixedValue(prefix, prop)
        self._delPrefixedValue(prefix, prop)
        return value

    def delProperty(self, prop, prefix=None):
        self._delPrefixedValue(prefix, prop)

    def _getPrefixedValue(self, prefix, name=None):
        if name is None:
            return self.system[f'{prefix}'] if prefix else self.system
        else:
            return self.system[f'{prefix}.{name}'] if prefix else self.system[name]

    def _setPrefixedValue(self, prefix, value, name=None):
        if name is None:
            if prefix:
                self.system[f'{prefix}'] = value
            else:
                self.system = value
        else:
            if prefix:
                self.system[f'{prefix}.{name}'] = value
            else:
                self.system[name] = value

    def _delPrefixedValue(self, prefix, name=None):
        if name is None:
            if prefix:
                del self.system[f'{prefix}']
            else:
                raise RuntimeError('Both prefix and name are None.')
        else:
            if prefix:
                del self.system[f'{prefix}.{name}']
            else:
                del self.system[name]

    def __getitem__(self, item):
        return self.system[item]

    def __contains__(self, item):
        return item in self.system
