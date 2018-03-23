import asyncio
import pickle as pcl
from copy import copy

from . import getSubClassByName


class WorkerError(Exception):
    pass


class Worker(object):

    shortname = None
    workers = {}

    def __init__(self, loop, name):
        super().__init__()
        self.loop = loop
        self.name = name

    def sendTask(self, *args, **kwargs):
        data = list(args)
        if kwargs:
            data.append(kwargs)

        for dct in data:
            try:
                if 'name' in dct:
                    name = dct['name']
                    if name not in self.workers:
                        self.workers[name] = getSubClassByName(Worker, dct['type'])(self.loop, name, **dct['params'])
            except KeyError as e:
                print('Warning: KeyError: {}, data : {}'.format(e, data))
                continue

        for dct in data:
            try:
                if 'toWorker' in dct and dct['toWorker'] in self.workers:
                    asyncio.ensure_future(self.workers[dct['toWorker']].run(dct['fromWorker'],**dct['data']))
            except KeyError as e:
                print('Warning: KeyError: {}, data : {}'.format(e, data))
                continue

    async def run(self, fromWorker : str):
        pass

    ############################################
    # Code responds for serialization
    def __getstate__(self):
        state = copy(self.__dict__)
        del state['loop']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.loop = asyncio.get_event_loop()


    @classmethod
    def save(cls):
        with open('workers.dump', 'wb') as f:
            pcl.dump(cls.workers, f)

    @classmethod
    def load(cls, loop):
        with open('workers.dump', 'rb') as f:
            cls.workers = pcl.load(f)
        for worker in cls.workers.values():
            worker.loop = loop
