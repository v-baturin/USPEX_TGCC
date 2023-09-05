import pickle as pcl

from typing import Union
from sqlalchemy import MetaData, ForeignKey, Table, Column, Integer, Float, String, create_engine, insert, select, and_

metadata_obj = MetaData()
pool = Table(
    "flavours",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("sID", Integer, nullable=False),
    Column("name", String, nullable=False),
)
propertiesInt = Table(
    "propertiesInt",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("fID", ForeignKey("flavours.id"), nullable=False),
    Column("prop", String(30), nullable=False),
    Column("value", Integer, nullable=False),
)
propertiesFlt = Table(
    "propertiesFlt",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("fID", ForeignKey("flavours.id"), nullable=False),
    Column("prop", String(30), nullable=False),
    Column("value", Float, nullable=False),
)
propertiesStr = Table(
    "propertiesStr",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("fID", ForeignKey("flavours.id"), nullable=False),
    Column("prop", String(30), nullable=False),
    Column("value", String, nullable=False),
)
propertiesObj = Table(
    "propertiesObj",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("fID", ForeignKey("flavours.id"), nullable=False),
    Column("prop", String(30), nullable=False),
    Column("value", String, nullable=False),
)


class FlavourFactory:

    def __init__(self, extensions):
        self.extensions = extensions

    def __call__(self, **kwargs):
        return EntryFlavour(extensions=self.extensions, **kwargs)


class EntryFlavour:
    
    def __init__(self, extensions=None, **properties):
        self.extensions = extensions if extensions is not None else {}
        self._properties = properties
        self.ID = None

    def setID(self, ID: int):
        self.ID = ID

    def getFactory(self):
        return FlavourFactory(self.extensions)

    def getProperties(self):
        return self._properties

    def getProperty(self, prop, extension=''):
        if f'{extension}.{prop}' not in self._properties:
            if self.ID is not None:
                value = self.getPropertyBD(f'{extension}.{prop}')
                if value is not None:
                    self._properties[f'{extension}.{prop}'] = value
                    return self._properties[f'{extension}.{prop}']
            if extension in self.extensions:
                self._properties[f'{extension}.{prop}'] = getattr(self.extensions[extension], prop)(self)
            else:
                raise KeyError(f'Can not evaluate property {extension}.{prop} for {self._properties}.')
        return self._properties[f'{extension}.{prop}']

    def getPropertyBD(self, prop):
        stmt = select(propertiesInt.c.value).where(and_(propertiesInt.c.fID == self.ID, propertiesInt.c.prop == prop))
        with PoolEntry.engine.connect() as conn:
            rows = conn.execute(stmt).all()
        assert len(rows) <= 1
        if len(rows) == 1:
            return rows[0][0]
        stmt = select(propertiesFlt.c.value).where(and_(propertiesFlt.c.fID == self.ID, propertiesFlt.c.prop == prop))
        with PoolEntry.engine.connect() as conn:
            rows = conn.execute(stmt).all()
        assert len(rows) <= 1
        if len(rows) == 1:
            return rows[0][0]
        stmt = select(propertiesStr.c.value).where(and_(propertiesStr.c.fID == self.ID, propertiesStr.c.prop == prop))
        with PoolEntry.engine.connect() as conn:
            rows = conn.execute(stmt).all()
        assert len(rows) <= 1
        if len(rows) == 1:
            return rows[0][0]
        stmt = select(propertiesObj.c.value).where(and_(propertiesObj.c.fID == self.ID, propertiesObj.c.prop == prop))
        with PoolEntry.engine.connect() as conn:
            rows = conn.execute(stmt).all()
        assert len(rows) <= 1
        if len(rows) == 1:
            return pcl.loads(rows[0][0])
        return None

    def setProperty(self, prop, value, extension=''):
        self._properties[f'{extension}.{prop}'] = value
        if self.ID is not None and self.getPropertyBD(f'{extension}.{prop}') is None:
            self.setPropertyBD(f'{extension}.{prop}', value)

    def setPropertyBD(self, prop: str, value: Union[int, str]):
        with PoolEntry.engine.connect() as conn:
            if isinstance(value, int):
                conn.execute(insert(propertiesInt), [{"fID": self.ID, "prop": prop, "value": value}])
            elif isinstance(value, float):
                conn.execute(insert(propertiesFlt), [{"fID": self.ID, "prop": prop, "value": value}])
            elif isinstance(value, str):
                conn.execute(insert(propertiesStr), [{"fID": self.ID, "prop": prop, "value": value}])
            else:
                conn.execute(insert(propertiesObj), [{"fID": self.ID, "prop": prop, "value": pcl.dumps(value)}])
            conn.commit()

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

    @classmethod
    def createEngine(cls, filename):
        cls.engine = create_engine(f"sqlite+pysqlite:///{filename}")
        metadata_obj.create_all(cls.engine)

    def __init__(self, ID: int,  system: EntryFlavour):
        self.ID = ID
        self.originalID = None
        self.duplicates = []
        self.expressions = {}
        self.flavourFactory = system.getFactory()
        self._flavours = {}
        self.properties = {}
        self.addFlavour('origin', system)

    @property
    def flavours(self):
        with self.engine.connect() as conn:
            result = conn.execute(select(pool.c.id, pool.c.name).where(pool.c.sID == self.ID)).all()
        if len(result) != len(self._flavours):
            self._flavours = {}
            for fID, flavour in result:
                self._flavours[flavour] = self.flavourFactory()
                self._flavours[flavour].setID(fID)
        return self._flavours

    def addFlavour(self, name: str, flavour: EntryFlavour):
        # assert name not in self.flavours, f'Flavour {name} already in system {self.ID}'
        with self.engine.connect() as conn:
            result = conn.execute(insert(pool), [{"sID": self.ID, "name": name}])
            conn.commit()
        ID = result.inserted_primary_key[0]
        self.flavours[name] = flavour
        flavour.setID(ID)
        for prop, value in flavour.getProperties().items():
            flavour.setPropertyBD(prop, value)

    def getFlavour(self, name: str) -> EntryFlavour:
        return self.flavours[name]

    def getProperty(self, prop, extension='', suffix='origin'):
        return self.getFlavour(suffix).getProperty(prop, extension=extension)

    def setProperty(self, prop, value, extension='', suffix='origin'):
        if suffix not in self.flavours:
            flavour = self.flavourFactory()
            self.addFlavour(suffix, flavour)
        else:
            flavour = self.getFlavour(suffix)
        flavour.setProperty(prop, value, extension=extension)

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
