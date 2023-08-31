import os
import pickle as pcl

from typing import Union
from sqlalchemy import MetaData, Table, Column, Integer, String, create_engine, insert, update, select, and_


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

    def serialize(self):
        return pcl.dumps(self._properties)

    def getProperty(self, prop, extension=''):
        if f'{extension}.{prop}' not in self._properties:
            if extension in self.extensions:
                self._properties[f'{extension}.{prop}'] = getattr(self.extensions[extension], prop)(self)
            else:
                raise KeyError(f'Can not evaluate property {extension}.{prop} for {self._properties}.')
        return self._properties[f'{extension}.{prop}']

    def setProperty(self, prop, value, extension=''):
        self._properties[f'{extension}.{prop}'] = value
        if extension in self.extensions and hasattr(self.extensions[extension], 'set'):
            self._properties.update(self.extensions[extension].set(self._properties, prop, value))

    def delProperty(self, prop, extension=''):
        del self._properties[f'{extension}.{prop}']

    def __getitem__(self, item: str):
        extension, prop, *other = item.split('.')
        assert not other, f'Too complex property name {item}.'
        return self.getProperty(prop, extension=extension)

    def __contains__(self, item: str):
        return item in self._properties


class PoolEntry:

    engine = None
    filename = None
    pool = None

    @classmethod
    def createEngine(cls, filename):
        cls.filename = filename
        cls.engine = create_engine(f"sqlite+pysqlite:///{filename}")
        metadata_obj = MetaData()
        cls.pool = Table(
            "flavours",
            metadata_obj,
            Column("id", Integer, primary_key=True),
            Column("sID", Integer, nullable=False),
            Column("name", String, nullable=False),
            Column("content", String, nullable=False),
        )
        metadata_obj.create_all(cls.engine)

    @classmethod
    def cleanDB(cls):
        os.remove(cls.filename)

    def __init__(self, ID: int,  system: EntryFlavour):
        self.ID = ID
        self.originalID = None
        self.duplicates = []
        self.expressions = {}
        self.flavourFactory = system.getFactory()
        self.flavours = []
        self.addFlavour('origin', system)

    def addFlavour(self, name: str, flavour: EntryFlavour):
        assert name not in self.flavours, f'Flavour {name} already in system {self.ID}'
        self.flavours.append(name)
        with self.engine.connect() as conn:
            conn.execute(insert(self.pool), [{"sID": self.ID, "name": name, "content": flavour.serialize()}])
            conn.commit()

    def getFlavour(self, name: str) -> EntryFlavour:
        stmt = select(self.pool.c.content).where(and_(self.pool.c.sID == self.ID, self.pool.c.name == name))
        with self.engine.connect() as conn:
            result = conn.execute(stmt)
        rows = result.all()
        assert len(rows) == 1
        return self.flavourFactory(**pcl.loads(rows[0][0]))

    def setFlavour(self, name, flavour: EntryFlavour):
        content = flavour.serialize()
        stmt = update(self.pool).where(and_(self.pool.c.sID == self.ID, self.pool.c.name == name)).values(content=content)
        with self.engine.connect() as conn:
            conn.execute(stmt)
            conn.commit()

    def getProperty(self, prop, extension='', suffix='origin'):
        return self.getFlavour(suffix).getProperty(prop, extension=extension)

    def setProperty(self, prop, value, extension='', suffix='origin'):
        if suffix in self.flavours:
            flavour = self.getFlavour(suffix)
            flavour.setProperty(prop, value, extension=extension)
            self.setFlavour(suffix, flavour)
        else:
            flavour = self.flavourFactory()
            flavour.setProperty(prop, value, extension=extension)
            self.addFlavour(suffix, flavour)

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
            return suffix in self.flavours and f'{prefix}.{prop}' in self.getFlavour(suffix)
        else:
            raise KeyError(f'Property {item} is not valid.')
