import logging
import numpy as np
from sqlalchemy import ForeignKey, UniqueConstraint, Table, Column, Integer, Float, String, select, update, delete, and_
from sqlalchemy.dialects.sqlite import insert

from .Engine import Engine
from .Flavour import FlavourFactory, Flavour
from .Entry import Entry
from .Expression import Expression
from ..Expressions.ExpressionEvaluator import ExpressionEvaluator


logger = logging.getLogger(__name__)


pools = Table(
    "pools",
    Engine.metadata_obj,
    Column("id", Integer, primary_key=True),
)
poolMap = Table(
    "poolMap",
    Engine.metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("entryID", ForeignKey("systems.id"), nullable=False),
    Column("poolID", ForeignKey("pools.id"), nullable=False),
    UniqueConstraint('entryID', 'poolID'),
)


class Pool:

    def __init__(self, ID: int, flavourfactory: FlavourFactory, expressionExtensions, metric=None):
        self.ID = ID
        self.flavourFactory = flavourfactory
        self.metric = metric
        self.expressionExtensions = expressionExtensions
        self._cache = {}

    @staticmethod
    def newPool(flavourfactory: FlavourFactory, expressionExtensions, metric=None):
        with Engine.engine.connect() as conn:
            result = conn.execute(insert(pools), [{}])
            conn.commit()
        return Pool(result.inserted_primary_key[0], flavourfactory, expressionExtensions, metric)

    def createPool(self):
        return Pool.newPool(self.flavourFactory, self.expressionExtensions, self.metric)

    def __copy__(self):
        newPool = self.createPool()
        for ID in self.getIDs():
            newPool.addEntry(self.getEntry(ID))
        return newPool

    def __hash__(self):
        return hash(self.ID)

    def __getstate__(self):
        return dict(ID=self.ID,
                    flavourFactory=self.flavourFactory,
                    expressionExtensions=self.expressionExtensions,
                    metric=self.metric
                    )

    def __setstate__(self, state):
        self.ID = state['ID']
        self.flavourFactory = state['flavourFactory']
        self.expressionExtensions = state['expressionExtensions']
        self.metric = state['metric']
        self._cache = {}

    def newEntry(self, flavour: Flavour):
        """
        Assign ID to system.

        :type system:
        :param system: system to be labeled with ID.

        """
        entry = Entry.newEntry(flavour, self.metric)
        self.addEntry(entry)
        return entry.ID

    def addEntry(self, entry: Entry):
        self._cache[entry.ID] = entry
        with Engine.engine.connect() as conn:
            stmt = insert(poolMap).values(poolID=self.ID, entryID=entry.ID)
            stmt = stmt.on_conflict_do_nothing()
            conn.execute(stmt)
            conn.commit()

    def getIDs(self) -> list:
        with Engine.engine.connect() as conn:
            IDs = conn.execute(select(poolMap.c.entryID).where(poolMap.c.poolID == self.ID)).all()
        return np.asarray(IDs, dtype=int).flatten().tolist()

    def getEntry(self, ID: int):
        # assert ID in self.getIDs()
        if ID not in self._cache:
            self._cache[ID] = Entry.getEntry(ID, self.flavourFactory, self.metric)
        return self._cache[ID]

    def fronts(self, expression):
        # expression = applyPresetsRecursive(expression)
        entries = [self.getEntry(ID) for ID in self.getIDs()]
        values = [entry[expression] if isinstance(expression, str) else entry.getExpression(expression)
                  for entry in entries]
        return [[entries[ind] for ind in np.flatnonzero(values == value)] for value in np.unique(values)]

    def createExpression(self, expression):
        if isinstance(expression, str):
            return expression
        else:
            return Expression(expression, self)

    def evaluate(self, expression):
        ExpressionEvaluator.calculate(expression, self)
