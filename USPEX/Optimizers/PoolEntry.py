from typing import Union


class EntryFactory:

    def __init__(self, extensions):
        self.extensions = extensions

    def __call__(self, **kwargs):
        return PoolEntry(extensions=self.extensions, **kwargs)

class PoolEntry:

    def __init__(self, extensions=None, ID=None,  **system):
        self.ID = ID
        self.originalID = None
        self.duplicates = []
        self.system = dict(origin=system)
        self.expressions = {}
        self.extensions = extensions if extensions is not None else {}

    def getProperty(self, prop, prefix='', suffix='origin'):
        system = self.system[suffix]
        if f'{prefix}.{prop}' not in system:
            if prefix in self.extensions:
                system[f'{prefix}.{prop}'] = getattr(self.extensions[prefix], prop)(system)
            else:
                raise KeyError(f'Extension {prefix} is not set for {self}.')
        return system[f'{prefix}.{prop}']

    def setProperty(self, prop, value, prefix='', suffix='origin'):
        if suffix not in self.system:
            self.system[suffix] = {}
        system = self.system[suffix]
        system[f'{prefix}.{prop}'] = value
        if prefix in self.extensions and hasattr(self.extensions[prefix], 'set'):
            self.extensions[prefix].set(system, prop, value)

    def delProperty(self, prop, prefix='', suffix='origin'):
        if suffix in self.system:
            del self.system[suffix][f'{prefix}.{prop}']

    def setExpression(self, expression: tuple, value):
        if expression not in self.expressions:
            self.expressions[expression] = []
        self.expressions[expression].append(value)

    def __getitem__(self, item: Union[str, tuple]):
        if item == 'ID':
            return self.ID
        elif isinstance(item, tuple):
            return self.expressions[item][-1]
        elif isinstance(item, str):
            prefix, prop, suffix, *other = item.split('.')
            assert not other, f'Too complex property name {item}.'
            return self.getProperty(prop, prefix, suffix)
        else:
            raise KeyError(f'Property {item} is not valid.')

    def __contains__(self, item: Union[str, tuple]):
        if item == 'ID':
            return True
        elif isinstance(item, tuple):
            return item in self.expressions
        elif isinstance(item, str):
            prefix, prop, suffix, *other = item.split('.')
            assert not other, f'Too complex property name {item}.'
            return suffix in self.system and '.'.join((prefix, prop)) in self.system[suffix]
        else:
            raise KeyError(f'Property {item} is not valid.')
