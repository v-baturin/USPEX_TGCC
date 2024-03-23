from sqlalchemy import MetaData, create_engine


class Engine:

    engine = None
    metadata_obj = MetaData()

    @classmethod
    def createEngine(cls, filename):
        if filename is not None:
            cls.engine = create_engine(f"sqlite+pysqlite:///{filename}")
            cls.metadata_obj.create_all(cls.engine)
