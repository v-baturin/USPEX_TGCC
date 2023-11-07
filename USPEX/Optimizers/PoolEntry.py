import logging
import pickle as pcl
import numpy as np

from typing import Union
from sqlalchemy import MetaData, ForeignKey, Table, Column, Integer, Float, String, create_engine, insert, select, and_


logger = logging.getLogger(__name__)


metadata_obj = MetaData()
systems = Table(
    "systems",
    metadata_obj,
    Column("id", Integer, primary_key=True),
)
flavours = Table(
    "flavours",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("sID", ForeignKey("systems.id"), nullable=False),
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
pools = Table(
    "pools",
    metadata_obj,
    Column("id", Integer, primary_key=True),
)
poolMap = Table(
    "poolMap",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("entryID", ForeignKey("systems.id"), nullable=False),
    Column("poolID", ForeignKey("pools.id"), nullable=False),
)
expressions = Table(
    "expressions",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("poolID", ForeignKey("pools.id"), nullable=False),
    Column("name", String, nullable=False),
)
expressionsInt = Table(
    "expressionsInt",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("sID", ForeignKey("systems.id"), nullable=False),
    Column("eID", ForeignKey("expressions.id"), nullable=False),
    Column("value", Integer, nullable=False),
)
expressionsFlt = Table(
    "expressionsFlt",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("sID", ForeignKey("systems.id"), nullable=False),
    Column("eID", ForeignKey("expressions.id"), nullable=False),
    Column("value", Float, nullable=False),
)
expressionsStr = Table(
    "expressionsStr",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("sID", ForeignKey("systems.id"), nullable=False),
    Column("eID", ForeignKey("expressions.id"), nullable=False),
    Column("value", String, nullable=False),
)
expressionsObj = Table(
    "expressionsObj",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("sID", ForeignKey("systems.id"), nullable=False),
    Column("eID", ForeignKey("expressions.id"), nullable=False),
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
        self._propertiesCache = properties
        self.ID = None

    def __getstate__(self):
        return dict(ID=self.ID, extensions=self.extensions)

    def __setstate__(self, state):
        self.ID = state['ID']
        self.extensions = state['extensions']
        self._propertiesCache = {}

    def setID(self, ID: int):
        self.ID = ID

    def getFactory(self):
        return FlavourFactory(self.extensions)

    def getProperties(self):
        return self._propertiesCache

    def getProperty(self, prop, extension=''):
        if f'{extension}.{prop}' not in self._propertiesCache:
            if self.ID is not None:
                value = self.getPropertyBD(f'{extension}.{prop}')
                if value is not None:
                    self._propertiesCache[f'{extension}.{prop}'] = value
                    return self._propertiesCache[f'{extension}.{prop}']
            if extension in self.extensions:
                self._propertiesCache[f'{extension}.{prop}'] = getattr(self.extensions[extension], prop)(self)
            else:
                raise KeyError(f'Can not evaluate property {extension}.{prop} for {self._propertiesCache}.')
        return self._propertiesCache[f'{extension}.{prop}']

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
        self._propertiesCache[f'{extension}.{prop}'] = value
        if self.ID is not None and self.getPropertyBD(f'{extension}.{prop}') is None:
            self.setPropertyBD(f'{extension}.{prop}', value)

    def setPropertyBD(self, prop: str, value):
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
        del self._propertiesCache[f'{extension}.{prop}']

    def __getitem__(self, item: str):
        extension, prop, *other = item.split('.')
        assert not other, f'Too complex property name {item}.'
        return self.getProperty(prop, extension=extension)

    def __contains__(self, item: str):
        extension, prop, *other = item.split('.')
        assert not other, f'Too complex property name {item}.'
        try:
            self.getProperty(prop, extension=extension)
        except Exception:
            result = False
        else:
            result = True
        return result


class PoolEntry:

    engine = None

    @classmethod
    def createEngine(cls, filename):
        cls.engine = create_engine(f"sqlite+pysqlite:///{filename}")
        metadata_obj.create_all(cls.engine)

    def __init__(self, ID: int, flavourFactory: FlavourFactory):
        self.ID = ID
        self.flavourFactory = flavourFactory
        self._expressionsCache = {}
        self._flavours = {}
        self.originalID = None
        self.duplicates = []

    def __getstate__(self):
        return dict(ID=self.ID, flavourFactory=self.flavourFactory, flavours=self._flavours,
                    originalID=self.originalID, duplicates=self.duplicates)

    def __setstate__(self, state):
        self.ID = state['ID']
        self.flavourFactory = state['flavourFactory']
        self._flavours = state['flavours']
        self._expressionsCache = {}
        self.originalID = state['originalID']
        self.duplicates = state['duplicates']

    @staticmethod
    def newEntry(flavour: EntryFlavour):
        """
        Assign ID to system.

        :type system:
        :param system: system to be labeled with ID.

        """
        with PoolEntry.engine.connect() as conn:
            result = conn.execute(insert(systems), [{}])
            conn.commit()
        ID = result.inserted_primary_key[0]
        entry = PoolEntry(ID, flavour.getFactory())
        entry.addFlavour('origin', flavour)
        logger.info(f"Entry {entry.ID} successfully created by {entry['.howCome.origin']} operator"
                    f" from {entry['.parent.origin']} parents.")
        return entry

    @staticmethod
    def getEntry(ID: int, flavourFactory: FlavourFactory):
        with PoolEntry.engine.connect() as conn:
            result = conn.execute(select(systems).where(systems.c.id == ID)).all()
        assert result
        return PoolEntry(ID, flavourFactory)

    @property
    def flavours(self):
        with self.engine.connect() as conn:
            result = conn.execute(select(flavours.c.id, flavours.c.name).where(flavours.c.sID == self.ID)).all()
        if len(result) != len(self._flavours):
            self._flavours = {}
            for fID, flavour in result:
                self._flavours[flavour] = self.flavourFactory()
                self._flavours[flavour].setID(fID)
        return self._flavours

    def addFlavour(self, name: str, flavour: EntryFlavour):
        # assert name not in self.flavours, f'Flavour {name} already in system {self.ID}'
        with self.engine.connect() as conn:
            result = conn.execute(insert(flavours), [{"sID": self.ID, "name": name}])
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

    def setExpression(self, expression, value):
        self._expressionsCache[expression] = value
        if self.getExpressionBD(expression.ID) is None:
            self.setExpressionBD(expression.ID, value)

    def setExpressionBD(self, exprID: int, value):
        with PoolEntry.engine.connect() as conn:
            if isinstance(value, int):
                conn.execute(insert(expressionsInt), [{"sID": self.ID, "eID": exprID, "value": value}])
            elif isinstance(value, float):
                conn.execute(insert(expressionsFlt), [{"sID": self.ID, "eID": exprID, "value": value}])
            elif isinstance(value, str):
                conn.execute(insert(expressionsStr), [{"sID": self.ID, "eID": exprID, "value": value}])
            else:
                conn.execute(insert(expressionsObj), [{"sID": self.ID, "eID": exprID, "value": pcl.dumps(value)}])
            conn.commit()

    def getExpression(self, expression):
        if expression not in self._expressionsCache:
            self._expressionsCache[expression] = self.getExpressionBD(expression.ID)
        return self._expressionsCache[expression]

    def getExpressionBD(self, exprID):
        stmt = select(expressionsInt.c.value).where(and_(expressionsInt.c.sID == self.ID, expressionsInt.c.eID == exprID))
        with PoolEntry.engine.connect() as conn:
            rows = conn.execute(stmt).all()
        assert len(rows) <= 1
        if len(rows) == 1:
            return rows[0][0]
        stmt = select(expressionsFlt.c.value).where(and_(expressionsFlt.c.sID == self.ID, expressionsFlt.c.eID == exprID))
        with PoolEntry.engine.connect() as conn:
            rows = conn.execute(stmt).all()
        assert len(rows) <= 1
        if len(rows) == 1:
            return rows[0][0]
        stmt = select(expressionsStr.c.value).where(and_(expressionsStr.c.sID == self.ID, expressionsStr.c.eID == exprID))
        with PoolEntry.engine.connect() as conn:
            rows = conn.execute(stmt).all()
        assert len(rows) <= 1
        if len(rows) == 1:
            return rows[0][0]
        stmt = select(expressionsObj.c.value).where(and_(expressionsObj.c.sID == self.ID, expressionsObj.c.eID == exprID))
        with PoolEntry.engine.connect() as conn:
            rows = conn.execute(stmt).all()
        assert len(rows) <= 1
        if len(rows) == 1:
            return pcl.loads(rows[0][0])
        return None


    def __getitem__(self, item):
        if isinstance(item, str):
            if item == 'ID':
                return self.ID
            prefix, prop, suffix, *other = item.split('.')
            assert not other, f'Too complex property name {item}.'
            return self.getProperty(prop, prefix, suffix)
        return self.getExpression(item)

    def __contains__(self, item: Union[str, tuple]):
        if item == 'ID':
            return True
        elif isinstance(item, tuple):
            return item in self._expressionsCache
        elif isinstance(item, str):
            prefix, prop, suffix, *other = item.split('.')
            assert not other, f'Too complex property name {item}.'
            return suffix in self.flavours and f'{prefix}.{prop}' in self.getFlavour(suffix)
        else:
            raise KeyError(f'Property {item} is not valid.')


class Expression:

    def __init__(self, expression, pool):
        with PoolEntry.engine.connect() as conn:
            result = conn.execute(select(expressions.c.id).where(and_(expressions.c.poolID == pool.ID),
                                                            expressions.c.name == str(expression))).all()
        if not result:
            with PoolEntry.engine.connect() as conn:
                result = conn.execute(insert(expressions), [{"poolID": pool.ID, "name": str(expression)}])
                conn.commit()
            self.ID = result.inserted_primary_key[0]
        else:
            self.ID = result[0][0]
        self._expression = expression
        self._pool = pool

    def __hash__(self):
        return hash(self.ID)

    def __eq__(self, other):
        return self.ID == other.ID


class Pool:

    def __init__(self, ID: int, flavourfactory: FlavourFactory):
        self.ID = ID
        self.flavourFactory = flavourfactory
        self._cache = {}

    @staticmethod
    def createPool(flavourfactory: FlavourFactory):
        with PoolEntry.engine.connect() as conn:
            result = conn.execute(insert(pools), [{}])
            conn.commit()
        return Pool(result.inserted_primary_key[0], flavourfactory)

    def __hash__(self):
        return hash(self.ID)

    def __getstate__(self):
        return dict(ID=self.ID, flavourFactory=self.flavourFactory)

    def __setstate__(self, state):
        self.ID = state['ID']
        self.flavourFactory = state['flavourFactory']
        self._cache = {}

    def newEntry(self, flavour: EntryFlavour):
        """
        Assign ID to system.

        :type system:
        :param system: system to be labeled with ID.

        """
        entry = PoolEntry.newEntry(flavour)
        self.addEntry(entry)
        return entry.ID

    def addEntry(self, entry: PoolEntry):
        self._cache[entry.ID] = entry
        with PoolEntry.engine.connect() as conn:
            conn.execute(insert(poolMap), [{"poolID": self.ID, "entryID": entry.ID}])
            conn.commit()

    def getIDs(self):
        with PoolEntry.engine.connect() as conn:
            IDs = conn.execute(select(poolMap.c.entryID).where(poolMap.c.poolID == self.ID)).all()
        return np.asarray(IDs, dtype=int).flatten().tolist()

    def getEntry(self, ID: int):
        assert ID in self.getIDs()
        if ID not in self._cache:
            self._cache[ID] = PoolEntry.getEntry(ID, self.flavourFactory)
        return self._cache[ID]

    def fronts(self, expression):
        # expression = applyPresetsRecursive(expression)
        entries = [self.getEntry(ID) for ID in self.getIDs()]
        values = [entry[expression] if isinstance(expression, str) else entry.getExpression(expression)
                  for entry in entries]
        return [[entries[ind] for ind in np.flatnonzero(values == value)] for value in np.unique(values)]

    def createExpression(self, expression):
        return Expression(expression, self)
