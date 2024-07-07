import logging
import numpy as np
from sqlalchemy import ForeignKey, UniqueConstraint, Table, Column, Integer, Float, String, select, update, delete, and_
from sqlalchemy.dialects.sqlite import insert
from collections import Sequence

from .Engine import Engine
from .Flavour import FlavourFactory, Flavour
from .Pool import Pool


logger = logging.getLogger(__name__)


generations = Table(
    "generations",
    Engine.metadata_obj,
    Column("id", Integer, primary_key=True),
)
generationsMap = Table(
    "generationsMap",
    Engine.metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("poolID", ForeignKey("pools.id"), nullable=False),
    Column("poolName", String, nullable=False),
    Column("generationID", ForeignKey("generations.id"), nullable=False),
    UniqueConstraint('poolID', 'generationID'),
    UniqueConstraint('poolName', 'generationID'),
)


class Generations(Sequence):

    def __init__(self, flavourfactory: FlavourFactory, expressionExtensions: dict[str, object], metric=None):
        self.flavourFactory = flavourfactory
        self.metric = metric
        self.expressionExtensions = expressionExtensions
        self._cache = {}

    def __getstate__(self) -> dict:
        return dict(flavourFactory=self.flavourFactory,
                    expressionExtensions=self.expressionExtensions,
                    metric=self.metric
                    )

    def __setstate__(self, state: dict):
        self.flavourFactory = state['flavourFactory']
        self.expressionExtensions = state['expressionExtensions']
        self.metric = state['metric']
        self._cache = {}

    def append(self, generation: dict[str, Pool]):
        with Engine.engine.connect() as conn:
            result = conn.execute(insert(generations), [{}])
            generationID = result.inserted_primary_key[0]
            for poolName, pool in generation.items():
                stmt = insert(generationsMap).values(generationID=generationID, poolID=pool.ID, poolName=poolName)
                stmt = stmt.on_conflict_do_nothing()
                conn.execute(stmt)
            conn.commit()
        self._cache[generationID] = generation

    def getGeneration(self, generationID) -> list[int]:
        if generationID not in self._cache:
            with Engine.engine.connect() as conn:
                pools = conn.execute(select(generationsMap.c.poolID, generationsMap.c.poolName).where(generationsMap.c.generationID == generationID)).all()
            generation = {}
            for poolID, poolName in pools:
                generation[poolName] = Pool(poolID, self.flavourFactory, self.expressionExtensions, self.metric)
            self._cache[generationID] = generation
        return self._cache[generationID]

    def __getitem__(self, item):
        with Engine.engine.connect() as conn:
            generationIDs = conn.execute(select(generations)).all()
        size = len(generationIDs)
        if size == 0:
            raise IndexError()
        generationIDs = np.asarray(generationIDs, dtype=int).flatten().tolist()
        assert size == max(generationIDs)
        if item < 0:
            item += size
        item += 1
        if item not in generationIDs:
            raise IndexError()
        return self.getGeneration(item)

    def __len__(self):
        with Engine.engine.connect() as conn:
            generationIDs = conn.execute(select(generations)).all()
        size = len(generationIDs)
        if size > 0:
            generationIDs = np.asarray(generationIDs, dtype=int).flatten().tolist()
            assert size == max(generationIDs)
        return size
