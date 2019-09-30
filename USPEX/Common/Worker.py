import pickle as pcl


class WorkerError(Exception):
    pass


class Worker(object):

    workers = []

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        cls.workers.append(instance)
        return instance


    # if not cls.workers:
    #     cls.loop.stop()

    @classmethod
    def save(cls):
        with open('workers.dump', 'wb') as f:
            pcl.dump(cls.workers, f)

    @classmethod
    def load(cls):
        with open('workers.dump', 'rb') as f:
            cls.workers = pcl.load(f)
