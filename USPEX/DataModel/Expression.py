import logging
from sqlalchemy import ForeignKey, UniqueConstraint, Table, Column, Integer, Float, String, select, update, delete, and_
from sqlalchemy.dialects.sqlite import insert

from .Engine import Engine


logger = logging.getLogger(__name__)


expressions = Table(
    "expressions",
    Engine.metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("poolID", ForeignKey("pools.id"), nullable=False),
    Column("name", String, nullable=False),
    UniqueConstraint('poolID', 'name'),
)


class Expression:

    def __init__(self, expression: tuple, poolID: int):
        with Engine.engine.connect() as conn:
            result = conn.execute(select(expressions.c.id).where(and_(expressions.c.poolID == poolID),
                                                            expressions.c.name == str(expression))).all()
        if not result:
            with Engine.engine.connect() as conn:
                result = conn.execute(insert(expressions), [{"poolID": poolID, "name": str(expression)}])
                conn.commit()
            self.ID = result.inserted_primary_key[0]
        else:
            self.ID = result[0][0]
        self._expression = expression
        self._poolID = poolID

    def __hash__(self) -> int:
        return hash(self.ID)

    def __eq__(self, other: 'Expression') -> bool:
        return self.ID == other.ID
