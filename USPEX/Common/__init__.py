import os
import logging
import pickle

# default name for file containing restore information
DEFAULT_FILENAME = 'dump'
logger = logging.getLogger(__name__)


def save(obj, filename=DEFAULT_FILENAME):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)
        logger.debug('Calculation written to a %s file' % filename)

def load(cls, filename=DEFAULT_FILENAME):
    with open(filename, 'rb') as fp:
        uspex = pickle.load(fp)
        assert isinstance(uspex, cls)
    logger.info('Calculation picked up from ' + filename)
    return uspex

def clean(filename=DEFAULT_FILENAME):
    if os.path.isfile(filename):
        os.remove(filename)

def get_all_subclasses(cls):
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses

def getSubClassByName(cls, name):
    assert isinstance(name, str)
    for subcls in get_all_subclasses(cls):
        if hasattr(subcls, 'shortname') and subcls.shortname == name:
            return subcls
    raise KeyError(name)

def dictifyList(elements : list):
    return [element.toDICT() for element in elements]
