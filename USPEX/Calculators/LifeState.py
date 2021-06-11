import os
from shutil import copyfile
import pickle as pcl


class LifeState:

    FILENAME = 'stages.dump'
    FILENAME_BACKUP = 'stages.dump.back'

    def __init__(self):
        self.systems = {}

    def save(self):
        if os.path.exists(LifeState.FILENAME):
            copyfile(LifeState.FILENAME, LifeState.FILENAME_BACKUP)
        with open(LifeState.FILENAME, 'wb') as f:
            pcl.dump(self, f)

    @staticmethod
    def load(population : list):
        if os.path.exists(LifeState.FILENAME):
            with open(LifeState.FILENAME, 'rb') as f:
                state = pcl.load(f)
            IDs = set(system['ID'] for system in population)
            if not set(state.systems.keys()) <= IDs:
                LifeState.clear()
                state = LifeState()
        else:
            state = LifeState()
        return state

    @staticmethod
    def clear():
        if os.path.exists(LifeState.FILENAME):
            os.remove(LifeState.FILENAME)
