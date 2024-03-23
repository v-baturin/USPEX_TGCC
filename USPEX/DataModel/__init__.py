from sqlalchemy import MetaData, create_engine


class DataModel:

    engine = create_engine('sqlite+pysqlite:///:memory:')
    metadata_obj = MetaData()

    @classmethod
    def createEngine(cls, filename):
        cls.engine = create_engine(f"sqlite+pysqlite:///{filename}")

    @classmethod
    def createTables(cls, *tables):
        cls.metadata_obj.create_all(cls.engine, tables)
