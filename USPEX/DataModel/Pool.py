import logging
import numpy as np
from typing import Union, Mapping
from sqlalchemy import ForeignKey, UniqueConstraint, Table, Column, Integer, select
from sqlalchemy.dialects.sqlite import insert

from .Engine import Engine
from .Flavour import FlavourFactory, Flavour
from .Entry import Entry
from .Expression import Expression
from ..Expressions.Functions import Functions


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

    def __init__(self, ID: int, flavourfactory: FlavourFactory, expressionExtensions: dict[str, object], metric=None):
        self.ID = ID
        self.flavourFactory = flavourfactory
        self.metric = metric
        self.expressionExtensions = expressionExtensions
        self._cache = {}

    @staticmethod
    def newPool(flavourfactory: FlavourFactory, expressionExtensions: dict[str, object], metric=None) -> 'Pool':
        with Engine.engine.connect() as conn:
            result = conn.execute(insert(pools), [{}])
            conn.commit()
        return Pool(result.inserted_primary_key[0], flavourfactory, expressionExtensions, metric)

    def createPool(self) -> 'Pool':
        return Pool.newPool(self.flavourFactory, self.expressionExtensions, self.metric)

    def __copy__(self) -> 'Pool':
        newPool = self.createPool()
        for ID in self.getIDs():
            newPool.addEntry(self.getEntry(ID))
        return newPool

    def __hash__(self) -> int:
        return hash(self.ID)

    def __getstate__(self) -> dict:
        return dict(ID=self.ID,
                    flavourFactory=self.flavourFactory,
                    expressionExtensions=self.expressionExtensions,
                    metric=self.metric
                    )

    def __setstate__(self, state: dict):
        self.ID = state['ID']
        self.flavourFactory = state['flavourFactory']
        self.expressionExtensions = state['expressionExtensions']
        self.metric = state['metric']
        self._cache = {}

    def newEntry(self, flavour: Flavour) -> int:
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

    def getIDs(self) -> list[int]:
        with Engine.engine.connect() as conn:
            IDs = conn.execute(select(poolMap.c.entryID).where(poolMap.c.poolID == self.ID)).all()
        return np.asarray(IDs, dtype=int).flatten().tolist()

    def getEntry(self, ID: int) -> 'Entry':
        # assert ID in self.getIDs()
        if ID not in self._cache:
            self._cache[ID] = Entry.getEntry(ID, self.flavourFactory, self.metric)
        return self._cache[ID]

    def fronts(self, expression: Union[Expression, str]) -> list[list['Entry']]:
        # expression = applyPresetsRecursive(expression)
        entries = [self.getEntry(ID) for ID in self.getIDs()]
        values = [entry[expression] if isinstance(expression, str) else entry.getExpression(expression)
                  for entry in entries]
        return [[entries[ind] for ind in np.flatnonzero(values == value)] for value in np.unique(values)]

    def createExpression(self, expression: Union[tuple, str]) -> Union[Expression, str]:
        if isinstance(expression, str):
            return expression
        else:
            return Expression(expression, self.ID)

    def evaluate(self, expression: Union[tuple, str]) -> np.ndarray:
        storedData = {}
        self._evaluate(expression, storedData)
        for expression, values in storedData.items():
            for ID, value in zip(self.getIDs(), values):
                if isinstance(expression, tuple):
                    self.getEntry(ID).setExpression(self.createExpression(expression), value)
        return storedData[expression]

    def _evaluate(self, expression: Union[str, tuple, int, float], storedData: dict) -> Union[np.ndarray, int, float]:
        if expression not in storedData:
            if len(self.getIDs()) == 0:
                valueArray = np.empty(0)
            elif isinstance(expression, tuple):
                funcName, *funcParams = expression
                assert isinstance(funcName, str), f'Incorrect type {type(funcName)} of function {funcName}.'
                arguments = [self._evaluate(param, storedData) for param in funcParams]
                size = min(len(arg) for arg in arguments if hasattr(arg, '__len__'))
                for i, arg in enumerate(arguments):
                    if hasattr(arg, '__len__'):
                        arguments[i] = arg[:size]
                funcName = funcName.split('.')
                if len(funcName) == 1:
                    valueArray = getattr(Functions, funcName[0])(*arguments)
                elif len(funcName) == 2:
                    extension, funcName = funcName
                    utility, expressionTable = self.expressionExtensions[extension]
                    valueArray = expressionTable[funcName](utility, *arguments)
                else:
                    raise RuntimeError(f"Too complex expression {'.'.join(expression)}.")
            elif isinstance(expression, str):
                value = [self.getEntry(ID)[expression] for ID in self.getIDs()]
                # value = [self.evaluateTerminal(expression, system) for system in self.pool]
                # unfortunately simple np.asarray spoils dictionaries
                if value and isinstance(value[0], Mapping):
                    valueArray = np.empty((len(value,)), dtype=type(value[0]))
                    for i, x in enumerate(value):
                        valueArray[i] = x
                else:
                    valueArray = np.asarray(value)
            else:
                # just a parameter. return it without doing anything.
                return expression
            storedData[expression] = valueArray
        return storedData[expression]
