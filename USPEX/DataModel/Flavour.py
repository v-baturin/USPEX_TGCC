import logging
import pickle as pcl
from sqlalchemy import ForeignKey, UniqueConstraint, Table, Column, Integer, Float, String, select, update, delete, and_
from sqlalchemy.dialects.sqlite import insert

from .Engine import Engine


logger = logging.getLogger(__name__)


flavours = Table(
    "flavours",
    Engine.metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("sID", ForeignKey("systems.id"), nullable=False),
    Column("name", String, nullable=False),
    UniqueConstraint('sID', 'name'),
)
propertiesInt = Table(
    "propertiesInt",
    Engine.metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("fID", ForeignKey("flavours.id"), nullable=False),
    Column("prop", String(30), nullable=False),
    Column("value", Integer, nullable=False),
    UniqueConstraint('fID', 'prop'),
)
propertiesFlt = Table(
    "propertiesFlt",
    Engine.metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("fID", ForeignKey("flavours.id"), nullable=False),
    Column("prop", String(30), nullable=False),
    Column("value", Float, nullable=False),
    UniqueConstraint('fID', 'prop'),
)
propertiesStr = Table(
    "propertiesStr",
    Engine.metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("fID", ForeignKey("flavours.id"), nullable=False),
    Column("prop", String(30), nullable=False),
    Column("value", String, nullable=False),
    UniqueConstraint('fID', 'prop'),
)
propertiesObj = Table(
    "propertiesObj",
    Engine.metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("fID", ForeignKey("flavours.id"), nullable=False),
    Column("prop", String(30), nullable=False),
    Column("value", String, nullable=False),
    UniqueConstraint('fID', 'prop'),
)


class FlavourFactory:

    def __init__(self, extensions, metric=None):
        self.extensions = extensions
        self.metric = metric

    def __call__(self, **kwargs):
        return Flavour(extensions=self.extensions, **kwargs)

    def fetchFlavours(self, entryID):
        with Engine.engine.connect() as conn:
            result = conn.execute(select(flavours.c.id, flavours.c.name).where(flavours.c.sID == entryID)).all()
        return {name: Flavour(ID=fID, extensions=self.extensions) for fID, name in result}


class Flavour:

    def __init__(self, ID=None, extensions=None, **properties):
        self.extensions = extensions if extensions is not None else {}
        self._propertiesCache = properties
        self.ID = ID

    def __getstate__(self):
        return dict(ID=self.ID, extensions=self.extensions)

    def __setstate__(self, state):
        self.ID = state['ID']
        self.extensions = state['extensions']
        self._propertiesCache = {}

    def setID(self, entryID: int, name: str):
        with Engine.engine.connect() as conn:
            result = conn.execute(insert(flavours), [{"sID": entryID, "name": name}])
            conn.commit()
        assert self.ID is None
        self.ID = result.inserted_primary_key[0]
        for prop, value in self._propertiesCache.items():
            self._setPropertyBD(prop, value)

    def getFactory(self):
        return FlavourFactory(self.extensions)

    def getProperty(self, prop, extension=''):
        if f'{extension}.{prop}' not in self._propertiesCache:
            if self.ID is not None:
                try:
                    self._propertiesCache[f'{extension}.{prop}'] = self._getPropertyBD(f'{extension}.{prop}')
                except KeyError as e:
                    logger.debug(e)
                else:
                    return self._propertiesCache[f'{extension}.{prop}']
            if extension == 'antiseeds':
                self._propertiesCache[f'{extension}.{prop}'] = 0.0
            elif extension in self.extensions:
                self._propertiesCache[f'{extension}.{prop}'] = getattr(self.extensions[extension], prop)(self)
            else:
                raise KeyError(f'Can not evaluate property {extension}.{prop} for {self._propertiesCache}.')
        return self._propertiesCache[f'{extension}.{prop}']

    def _getPropertyBD(self, prop):
        stmt = select(propertiesInt.c.value).where(and_(propertiesInt.c.fID == self.ID, propertiesInt.c.prop == prop))
        with Engine.engine.connect() as conn:
            rows = conn.execute(stmt).all()
        assert len(rows) <= 1
        if len(rows) == 1:
            return rows[0][0]
        stmt = select(propertiesFlt.c.value).where(and_(propertiesFlt.c.fID == self.ID, propertiesFlt.c.prop == prop))
        with Engine.engine.connect() as conn:
            rows = conn.execute(stmt).all()
        assert len(rows) <= 1
        if len(rows) == 1:
            return rows[0][0]
        stmt = select(propertiesStr.c.value).where(and_(propertiesStr.c.fID == self.ID, propertiesStr.c.prop == prop))
        with Engine.engine.connect() as conn:
            rows = conn.execute(stmt).all()
        assert len(rows) <= 1
        if len(rows) == 1:
            return rows[0][0]
        stmt = select(propertiesObj.c.value).where(and_(propertiesObj.c.fID == self.ID, propertiesObj.c.prop == prop))
        with Engine.engine.connect() as conn:
            rows = conn.execute(stmt).all()
        assert len(rows) <= 1
        if len(rows) == 1:
            return pcl.loads(rows[0][0])
        raise KeyError(f"Can't find property {prop} for flavour {self.ID} in the database.")

    def setProperty(self, prop, value, extension=''):
        self._propertiesCache[f'{extension}.{prop}'] = value
        if self.ID is not None:
            self._setPropertyBD(f'{extension}.{prop}', value)

    def _setPropertyBD(self, prop: str, value):
        with Engine.engine.connect() as conn:
            if isinstance(value, int):
                stmt = insert(propertiesInt).values(fID=self.ID, prop=prop, value=value)
                stmt = stmt.on_conflict_do_update(index_elements=['fID', 'prop'], set_=dict(value=value))
                conn.execute(stmt)
            elif isinstance(value, float):
                stmt = insert(propertiesFlt).values(fID=self.ID, prop=prop, value=value)
                stmt = stmt.on_conflict_do_update(index_elements=['fID', 'prop'], set_=dict(value=value))
                conn.execute(stmt)
            elif isinstance(value, str):
                stmt = insert(propertiesStr).values(fID=self.ID, prop=prop, value=value)
                stmt = stmt.on_conflict_do_update(index_elements=['fID', 'prop'], set_=dict(value=value))
                conn.execute(stmt)
            else:
                stmt = insert(propertiesObj).values(fID=self.ID, prop=prop, value=pcl.dumps(value))
                stmt = stmt.on_conflict_do_update(index_elements=['fID', 'prop'], set_=dict(value=pcl.dumps(value)))
                conn.execute(stmt)
            conn.commit()

    def _updatePropertyBD(self, prop: str, value):
        with Engine.engine.connect() as conn:
            if isinstance(value, int):
                conn.execute(update(propertiesInt).where(
                    and_(propertiesInt.c.fID == self.ID, propertiesInt.c.prop == prop)).values(value=value))
            elif isinstance(value, float):
                conn.execute(update(propertiesFlt).where(
                    and_(propertiesFlt.c.fID == self.ID, propertiesFlt.c.prop == prop)).values(value=value))
            elif isinstance(value, str):
                conn.execute(update(propertiesStr).where(
                    and_(propertiesStr.c.fID == self.ID, propertiesStr.c.prop == prop)).values(value=value))
            else:
                conn.execute(update(propertiesObj).where(
                    and_(propertiesObj.c.fID == self.ID, propertiesObj.c.prop == prop)).values(value=pcl.dumps(value)))
            conn.commit()

    def delProperty(self, prop, extension=''):
        if f'{extension}.{prop}' in self._propertiesCache:
            del self._propertiesCache[f'{extension}.{prop}']
        if self.ID is not None:
            try:
                value = self._getPropertyBD(f'{extension}.{prop}')
            except KeyError:
                pass
            else:
                self._delPropertyBD(f'{extension}.{prop}', value)

    def _delPropertyBD(self, prop, value):
        with Engine.engine.connect() as conn:
            if isinstance(value, int):
                conn.execute(
                    delete(propertiesInt).where(and_(propertiesInt.c.fID == self.ID, propertiesInt.c.prop == prop)))
            elif isinstance(value, float):
                conn.execute(
                    delete(propertiesFlt).where(and_(propertiesFlt.c.fID == self.ID, propertiesFlt.c.prop == prop)))
            elif isinstance(value, str):
                conn.execute(
                    delete(propertiesStr).where(and_(propertiesStr.c.fID == self.ID, propertiesStr.c.prop == prop)))
            else:
                conn.execute(
                    delete(propertiesObj).where(and_(propertiesObj.c.fID == self.ID, propertiesObj.c.prop == prop)))
            conn.commit()

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
