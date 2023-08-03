from typing import Union


class FlavourFactory:

    def __init__(self, extensions):
        self.extensions = extensions

    def __call__(self, **kwargs):
        return EntryFlavour(extensions=self.extensions, **kwargs)


class EntryFlavour:
    
    def __init__(self, extensions=None, **properties):
        self.extensions = extensions if extensions is not None else {}
        self._properties = properties

    def getFactory(self):
        return FlavourFactory(self.extensions)

    def getProperty(self, prop, extension=''):
        if f'{extension}.{prop}' not in self._properties:
            if extension in self.extensions:
                self._properties[f'{extension}.{prop}'] = getattr(self.extensions[extension], prop)(self._properties)
            else:
                raise KeyError(f'Can not evaluate property {extension}.{prop} for {self._properties}.')
        return self._properties[f'{extension}.{prop}']

    def setProperty(self, prop, value, extension=''):
        self._properties[f'{extension}.{prop}'] = value
        if extension in self.extensions and hasattr(self.extensions[extension], 'set'):
            self.extensions[extension].set(self._properties, prop, value)

    def delProperty(self, prop, extension=''):
        del self._properties[f'{extension}.{prop}']

    def __getitem__(self, item: str):
        extension, prop, *other = item.split('.')
        assert not other, f'Too complex property name {item}.'
        return self.getProperty(prop, extension=extension)

    def __contains__(self, item: str):
        return item in self._properties


class PoolEntry:

    def __init__(self, ID,  system: EntryFlavour):
        self.ID = ID
        self.originalID = None
        self.duplicates = []
        self.expressions = {}
        self.flavourFactory = system.getFactory()
        self.system = dict(origin=system)

    def getProperty(self, prop, extension='', suffix='origin'):
        return self.system[suffix].getProperty(prop, extension=extension)

    def setProperty(self, prop, value, extension='', suffix='origin'):
        if suffix not in self.system:
            self.system[suffix] = self.flavourFactory()
        self.system[suffix].setProperty(prop, value, extension=extension)

    def delProperty(self, prop, extension='', suffix='origin'):
        if suffix in self.system:
            self.system[suffix].delProperty(prop, extension=extension)

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
            return suffix in self.system and f'{prefix}.{prop}' in self.system[suffix]
        else:
            raise KeyError(f'Property {item} is not valid.')
