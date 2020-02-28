'''
@file        Config.py
@author:     Artem Samtsevich
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    samtsevichartem@gmail.com
@date        December 2017
@brief       Abstract class for engines of calculations in USPEX, like USPEX, VCNEB ...
'''


class Config(object):


    def __init__(self, **kwargs):
        super().__init__()

    def isGoodSystem(self, system):
        return True

    @property
    def systemFactory(self):
        return None