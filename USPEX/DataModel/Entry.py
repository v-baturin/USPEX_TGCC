import logging
import pickle as pcl
from typing import Union
from sqlalchemy import ForeignKey, UniqueConstraint, Table, Column, Integer, Float, String, select, update, delete, and_
from sqlalchemy.dialects.sqlite import insert

from .Engine import Engine
from .Flavour import Flavour, FlavourFactory
from .Expression import Expression


logger = logging.getLogger(__name__)


systems = Table(
    "systems",
    Engine.metadata_obj,
    Column("id", Integer, primary_key=True),
)
expressionsInt = Table(
    "expressionsInt",
    Engine.metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("sID", ForeignKey("systems.id"), nullable=False),
    Column("eID", ForeignKey("expressions.id"), nullable=False),
    Column("value", Integer, nullable=False),
    UniqueConstraint('sID', 'eID'),
)
expressionsFlt = Table(
    "expressionsFlt",
    Engine.metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("sID", ForeignKey("systems.id"), nullable=False),
    Column("eID", ForeignKey("expressions.id"), nullable=False),
    Column("value", Float, nullable=False),
    UniqueConstraint('sID', 'eID'),
)
expressionsStr = Table(
    "expressionsStr",
    Engine.metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("sID", ForeignKey("systems.id"), nullable=False),
    Column("eID", ForeignKey("expressions.id"), nullable=False),
    Column("value", String, nullable=False),
    UniqueConstraint('sID', 'eID'),
)
expressionsObj = Table(
    "expressionsObj",
    Engine.metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("sID", ForeignKey("systems.id"), nullable=False),
    Column("eID", ForeignKey("expressions.id"), nullable=False),
    Column("value", String, nullable=False),
    UniqueConstraint('sID', 'eID'),
)

class Entry:

    def __init__(self, ID: int, flavourFactory: FlavourFactory, metric=None):
        self.ID = ID
        self.flavourFactory = flavourFactory
        self.metric = metric
        self._expressionsCache = {}
        self._flavours = {}

    def __getstate__(self) -> dict:
        return dict(ID=self.ID, flavourFactory=self.flavourFactory, metric=self.metric)

    def __setstate__(self, state: dict):
        self.ID = state['ID']
        self.flavourFactory = state['flavourFactory']
        self.metric = state['metric']
        self._flavours = {}
        self._expressionsCache = {}

    @staticmethod
    def newEntry(flavour: Flavour, metric=None) -> 'Entry':
        """
        Assign ID to system.

        :type system:
        :param system: system to be labeled with ID.

        """
        with Engine.engine.connect() as conn:
            result = conn.execute(insert(systems), [{}])
            conn.commit()
        ID = result.inserted_primary_key[0]
        entry = Entry(ID, flavour.getFactory(), metric)
        entry.addFlavour('origin', flavour)
        logger.info(f"Entry {entry.ID} successfully created by {entry['.howCome.origin']} operator"
                    f" from {entry['.parent.origin']} parents.")
        return entry

    @staticmethod
    def getEntry(ID: int, flavourFactory: FlavourFactory, metric=None) -> 'Entry':
        with Engine.engine.connect() as conn:
            result = conn.execute(select(systems).where(systems.c.id == ID)).all()
        assert result
        return Entry(ID, flavourFactory, metric)

    @property
    def flavours(self) -> dict[str, Flavour]:
        if not self._flavours:
            self._flavours = self.flavourFactory.fetchFlavours(self.ID)
        return self._flavours

    def addFlavour(self, name: str, flavour: Flavour):
        # assert name not in self.flavours, f'Flavour {name} already in system {self.ID}'
        flavour.setID(self.ID, name)
        self._flavours = {}

    def getFlavour(self, name: str) -> Flavour:
        return self.flavours[name]

    def getProperty(self, prop: str, extension: str = '', suffix: str = 'origin'):
        return self.getFlavour(suffix).getProperty(prop, extension=extension)

    def setProperty(self, prop: str, value, extension: str = '', suffix: str = 'origin'):
        if suffix not in self.flavours:
            flavour = self.flavourFactory()
            self.addFlavour(suffix, flavour)
        else:
            flavour = self.getFlavour(suffix)
        flavour.setProperty(prop, value, extension=extension)

    def delProperty(self, prop: str, extension: str = '', suffix: str = 'origin'):
        return self.getFlavour(suffix).delProperty(prop, extension=extension)

    def setExpression(self, expression: Expression, value):
        self._expressionsCache[expression] = value
        if self.getExpressionBD(expression.ID) is None:
            self.setExpressionBD(expression.ID, value)

    def setExpressionBD(self, exprID: int, value):
        with Engine.engine.connect() as conn:
            if isinstance(value, int):
                stmt = insert(expressionsInt).values(sID=self.ID, eID=exprID, value=value)
                stmt = stmt.on_conflict_do_update(index_elements=['sID', 'eID'], set_=dict(value=value))
                conn.execute(stmt)
            elif isinstance(value, float):
                stmt = insert(expressionsFlt).values(sID=self.ID, eID=exprID, value=value)
                stmt = stmt.on_conflict_do_update(index_elements=['sID', 'eID'], set_=dict(value=value))
                conn.execute(stmt)
            elif isinstance(value, str):
                stmt = insert(expressionsStr).values(sID=self.ID, eID=exprID, value=value)
                stmt = stmt.on_conflict_do_update(index_elements=['sID', 'eID'], set_=dict(value=value))
                conn.execute(stmt)
            else:
                stmt = insert(expressionsObj).values(sID=self.ID, eID=exprID, value=pcl.dumps(value))
                stmt = stmt.on_conflict_do_update(index_elements=['sID', 'eID'], set_=dict(value=pcl.dumps(value)))
                conn.execute(stmt)
            conn.commit()

    def getExpression(self, expression: Expression):
        if expression not in self._expressionsCache:
            self._expressionsCache[expression] = self.getExpressionBD(expression.ID)
        return self._expressionsCache[expression]

    def getExpressionBD(self, exprID: int):
        stmt = select(expressionsInt.c.value).where(and_(expressionsInt.c.sID == self.ID, expressionsInt.c.eID == exprID))
        with Engine.engine.connect() as conn:
            rows = conn.execute(stmt).all()
        assert len(rows) <= 1
        if len(rows) == 1:
            return rows[0][0]
        stmt = select(expressionsFlt.c.value).where(and_(expressionsFlt.c.sID == self.ID, expressionsFlt.c.eID == exprID))
        with Engine.engine.connect() as conn:
            rows = conn.execute(stmt).all()
        assert len(rows) <= 1
        if len(rows) == 1:
            return rows[0][0]
        stmt = select(expressionsStr.c.value).where(and_(expressionsStr.c.sID == self.ID, expressionsStr.c.eID == exprID))
        with Engine.engine.connect() as conn:
            rows = conn.execute(stmt).all()
        assert len(rows) <= 1
        if len(rows) == 1:
            return rows[0][0]
        stmt = select(expressionsObj.c.value).where(and_(expressionsObj.c.sID == self.ID, expressionsObj.c.eID == exprID))
        with Engine.engine.connect() as conn:
            rows = conn.execute(stmt).all()
        assert len(rows) <= 1
        if len(rows) == 1:
            return pcl.loads(rows[0][0])
        return None


    def __getitem__(self, item: Union[str, Expression]):
        if isinstance(item, str):
            if item == 'ID':
                return self.ID
            prefix, prop, suffix, *other = item.split('.')
            assert not other, f'Too complex property name {item}.'
            return self.getProperty(prop, prefix, suffix)
        return self.getExpression(item)

    def __contains__(self, item: Union[str, Expression]) -> bool:
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

    def __eq__(self, other: 'Entry') -> bool:
        if self.metric is None:
            raise RuntimeError("Metric is not defined.")
        else:
            return self.metric.equal(self, other)
