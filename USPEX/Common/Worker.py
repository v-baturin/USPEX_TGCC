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

    @staticmethod
    def save():
        with open('workers.dump', 'wb') as f:
            pcl.dump(Worker.workers, f)

    @staticmethod
    def load():
        with open('workers.dump', 'rb') as f:
            Worker.workers = pcl.load(f)
