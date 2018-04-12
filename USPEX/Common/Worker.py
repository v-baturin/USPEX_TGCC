import asyncio
import pickle as pcl
from copy import copy

from . import getSubClassByName


class WorkerError(Exception):
    pass


class Worker(object):

    shortname = None
    workers = {}
    loop = None

    def __init__(self, name):
        super().__init__()
        self.name = name

    @classmethod
    def sendTask(cls, *args, **kwargs):
        data = list(args)
        if kwargs:
            data.append(kwargs)

        for dct in data:
            try:
                if 'name' in dct:
                    name = dct['name']
                    if 'kill' in dct:
                        del cls.workers[name]
                    elif name not in cls.workers:
                        cls.workers[name] = getSubClassByName(Worker, dct['type'])(name, **dct['params'])
            except KeyError as e:
                print('Warning: KeyError: {}, data : {}'.format(e, data))
                continue

        for dct in data:
            try:
                if 'toWorker' in dct and dct['toWorker'] in cls.workers:
                    asyncio.ensure_future(cls.workers[dct['toWorker']].run(dct['fromWorker'],**dct['data']))
            except KeyError as e:
                print('Warning: KeyError: {}, data : {}'.format(e, data))
                continue

        if not cls.workers:
            cls.loop.stop()

    async def run(self, fromWorker : str):
        pass

    @classmethod
    def save(cls):
        with open('workers.dump', 'wb') as f:
            pcl.dump(cls.workers, f)

    @classmethod
    def load(cls):
        with open('workers.dump', 'rb') as f:
            cls.workers = pcl.load(f)
