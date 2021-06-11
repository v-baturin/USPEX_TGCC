import logging
logger = logging.getLogger(__name__)


# -*- coding:utf-8 -*-
#
# This program is free software. It comes without any warranty, to
# the extent permitted by applicable law. You can redistribute it
# and/or modify it under the terms of the GNU Public License 2 or upper.
# Please ask if you wish a more permissive license.

import asyncio


from subprocess import getoutput
from os.path import getsize
from time import sleep


from Calculators.Calculators.Common.Connector import Connector


class ScreenNotFoundError(Exception):
    """raised when the screen does not exists"""
    def __init__(self, message, screen_name):
        message += " Screen \"{0}\" not found".format(screen_name)
        self.screen_name = screen_name
        super(ScreenNotFoundError, self).__init__(message)


def tailf(file_):
    """Each value is content added to the log file since last value return"""
    last_size = getsize(file_)
    while True:
        cur_size = getsize(file_)
        if (cur_size != last_size):
            f = open(file_, 'r')
            f.seek(last_size if cur_size > last_size else 0)
            text = f.read()
            f.close()
            last_size = cur_size
            yield text
        else:
            yield ""


def _exec(cmd : str, connector=None):
    if connector is not None:
        loop = asyncio.get_event_loop()
        task = asyncio.ensure_future(connector.execute(cmd), loop=loop)
        res = loop.run_until_complete(task).stdout
        loop.close()
    else:
        res = getoutput(cmd)
    return res


def list_screens(connector=None):
    """List all the existing screens and build a Screen instance for each
    """
    list_cmd = "screen -ls"
    lines = _exec(list_cmd, connector=connector).split('\n')
    return [
                Screen(".".join(l.split(".")[1:]).split("\t")[0])
                for l in lines
                if "\t" in l and ".".join(l.split(".")[1:]).split("\t")[0]
            ]


class Screen(object):
    """Represents a gnu-screen object::

        >>> s=Screen("screenName", initialize=True)
        >>> s.name
        'screenName'
        >>> s.exists
        True
        >>> s.state
        >>> s.send_commands("man -k keyboard")
        >>> s.kill()
        >>> s.exists
        False
    """

    def __init__(self, name, initialize=False, connector : Connector =None):
        self.name = name
        self._id = None
        self._status = None
        self.logs = None
        self._logfilename = None
        self._connector = connector
        if initialize:
            self.initialize()

    @property
    def id(self):
        """return the identifier of the screen as string"""
        if not self._id:
            self._set_screen_infos()
        return self._id

    @property
    def status(self):
        """return the status of the screen as string"""
        self._set_screen_infos()
        return self._status

    @property
    def exists(self):
        """Tell if the screen session exists or not."""
        # Parse the screen -ls call, to find if the screen exists or not.
        #  "	28062.G.Terminal	(Detached)"
        lines = _exec('screen -ls', self._connector).split('\n')
        # lines = getoutput("screen -ls").split('\n')
        return self.name in [".".join(l.split(".")[1:]).split("\t")[0]
                             for l in lines if self.name in l]

    def enable_logs(self, filename=None):
        if filename is None:
            filename = self.name
        self._screen_commands("logfile " + filename, "log on")
        self._logfilename = filename
        open(filename, 'w+')
        self.logs = tailf(filename)

    def disable_logs(self, remove_logfile=False):
        self._screen_commands("log off")
        if remove_logfile:
            _exec('rm ' + self._logfilename)
            # system('rm ' + self._logfilename)
        self.logs = None

    def initialize(self):
        """initialize a screen, if does not exists yet"""
        if not self.exists:
            self._id = None
            # CMD = 'screen -UR ' + self.name
            CMD = 'screen -dmS ' + self.name
            _exec(CMD, self._connector)
            print('Screen {} has been intialized.'.format(self.name))
            # if self._connector:
            #     loop = asyncio.get_event_loop()
            #     loop.run_until_complete(asyncio.ensure_future(self._connector.execute(CMD), loop=loop))
            #     loop.close()
            # else:
            #     system(CMD)

    def interrupt(self):
        """Insert CTRL+C in the screen session"""
        self._screen_commands("eval \"stuff \\003\"")

    def kill(self):
        """Kill the screen applications then close the screen"""
        self._screen_commands('quit')

    def detach(self):
        """detach the screen"""
        self._check_exists()
        _exec("screen -d " + self.id)
        # system("screen -d " + self.id)

    def send_commands(self, *commands):
        """send commands to the active gnu-screen"""
        self._check_exists()
        for command in commands:
            self._screen_commands('stuff "' + command + '" ',
                                  'eval "stuff \\015"')

    def add_user_access(self, unix_user_name):
        """allow to share your session with an other unix user"""
        self._screen_commands('multiuser on', 'acladd ' + unix_user_name)

    def _screen_commands(self, *commands):
        """allow to insert generic screen specific commands
        a glossary of the existing screen command in `man screen`"""
        self._check_exists()
        for command in commands:
            _exec('screen -x ' + self.id + ' -p 0 -X ' + command) # + "`echo -ne '\015'`")
            # system('screen -x ' + self.id + ' -X ' + command)
            sleep(0.02)

    def _check_exists(self, message="Error code: 404."):
        """check whereas the screen exist. if not, raise an exception"""
        if not self.exists:
            raise ScreenNotFoundError(message, self.name)

    def _set_screen_infos(self):
        """set the screen information related parameters"""
        if self.exists:
            lines = _exec('screen -ls', self._connector).split('\n')
            line = ""
            for l in lines:
                if (
                        l.startswith('\t') and
                        self.name in l and
                        self.name == ".".join(
                            l.split('\t')[1].split('.')[1:]) in l):
                    line = l
            if not line:
                raise ScreenNotFoundError("While getting info.", self.name)
            infos = line.split('\t')[1:]
            self._id = infos[0].split('.')[0]
            if len(infos) == 3:
                self._date = infos[1][1:-1]
                self._status = infos[2][1:-1]
            else:
                self._status = infos[1][1:-1]

    def _delayed_detach(self):
        sleep(0.5)
        self.detach()

    def __repr__(self):
        return "<%s '%s'>" % (self.__class__.__name__, self.name)
