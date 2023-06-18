import numpy as np

from ..Fitness.presets import applyPresets


class AtomisticPoolEntry:

    structureType = None
    atomType = None
    cellType = None
    atomicDisassemblerType = None

    @classmethod
    def registerTypes(cls, structureType, atomType, cellType, atomicDisassemblerType):
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType
        cls.atomicDisassemblerType = atomicDisassemblerType

    def __init__(self, **system):
        self.system = system
        self.expressions = {}

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

    def setExpression(self, expression, value):
        if expression not in self.expressions:
            self.expressions[expression] = []
        self.expressions[expression].append(value)

    def __getitem__(self, item):
        if item in self.system:
            value = self.system[item]
        elif item in self.expressions:
            value = self.expressions[item][-1]
        else:
            raise KeyError(f'Property {item} is not set.')
        return value

    def __contains__(self, item):
        return item in self.system

    @staticmethod
    def fronts(pool, expression):
        expression = applyPresets(expression)
        values = [s[expression] for s in pool]
        return [[pool[ind] for ind in np.flatnonzero(values == value)] for value in np.unique(values)]

