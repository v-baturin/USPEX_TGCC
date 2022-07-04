"""
USPEX.Stages.TaskManagers.Screen
================================

.. codeauthor:: Artem Samtsevich <samtsevichartem@gmail.com>

"""

import logging
import asyncio
from os.path import getsize

logger = logging.getLogger(__name__)


class ScreenNotFoundError(Exception):
    """raised when the screen does not exists"""
    def __init__(self, message, screen_name):
        message += f" Screen \"{screen_name}\" not found"
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

# def _exec(cmd : str, connector=None):
#     if connector is not None:
#         loop = asyncio.get_event_loop()
#         task = asyncio.ensure_future(connector.execute(cmd), loop=loop)
#         res = loop.run_until_complete(task).stdout
#         loop.close()
#     else:
#         res = getoutput(cmd)
#     return res

async def kill_screen_by_id(connector, id : int):
    await _exec(f'screen -dm -x {id} -p 0 -X  {"quit"}', connector=connector)
    await asyncio.sleep(0.02)

async def _exec(cmd, connector, cwd='.'):
    returncode, out, err = await connector.execute(cmd, cwd=cwd)
    await asyncio.sleep(0.1)
    logger.debug(f'process returned code {returncode}')
    if returncode != 0:
        logger.error(err)
        logger.error(out)
    return out

async def list_screens_id(connector):
    list_cmd = "screen -ls "
    lines = (await _exec(list_cmd, connector=connector)).split('\n')
    return [int(l.split(".")[0]) for l in lines if "\t" in l and ".".join(l.split(".")[1:]).split("\t")[0]]


async def list_screens(connector):
    """
    List all the existing screens and build a Screen instance for each
    """
    list_cmd = "screen -ls "
    res = await _exec(list_cmd, connector=connector)
    lines = res.split('\n')
    # lines = await _exec(list_cmd, connector=connector).split('\n')
    screens = [
                Screen(name=".".join(l.split(".")[1:]).split("\t")[0], connector=connector)
                for l in lines
                if "\t" in l and ".".join(l.split(".")[1:]).split("\t")[0]
              ]
    # if initialize:
    #     for scrn in screens:
    #         await scrn.initialize()
    return screens


class Screen(object):
    '''
    Represents a gnu-screen object::

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
    '''

    def __init__(self, name : str, connector):
        self.name = name
        self._id = None
        self._status = None
        self.logs = None
        self._logfilename = None
        self._connector = connector

    async def run_and_exit(self, command : str, cwd='.') -> int:
        '''
        Takes just a command and run only it and then it finishes and exit.
        :param command:
        :param cwd:
        :return: ID of this calculation
        '''
        script = f"""
        screen -S {self.name} -p 0 -dm bash -c 'echo "$STY" > jobID && {command}'
        while [ ! -f ./jobID ]
        do
          sleep 0.2
        done
        cat jobID
        rm jobID
        """
        out = await _exec(script, connector=self._connector, cwd=cwd)
        # returncode, out, err = await self._connector.execute('bash', input=script, cwd=cwd)
        return int(out.split('.')[0])

    async def initialize(self, cwd='.'):
        """initialize a screen, if does not exists yet"""
        # self.lock = asyncio.Lock()
        print('Initialization starts')
        # if not (await self.exists):
        self._id = None
        # await self.lock.acquire()
        # Detach the screen once attached, on a new tread.
        # Thread(target=self._delayed_detach).start()
        # support Unicode (-U),
        # attach to a new/existing named screen (-R).
        # getoutput('screen -UR ' + self.name)
        # cmd = 'screen -UR ' + self.name
        cmd = f'screen -mS {self.name}'
        # cmd = 'screen -dmS ' + self.name
        # res = check_output(cmd, shell=True)
        res = await _exec(cmd, connector=self._connector, cwd=cwd)
        await asyncio.sleep(5)
        out = await self.detach()
        # await self.lock.release()
        print('Initialization ends')

    @property
    async def id(self):
        """return the identifier of the screen as string"""
        if not self._id:
            await self._set_screen_infos()
        return self._id

    @property
    async def status(self):
        """return the status of the screen as string"""
        await self._set_screen_infos()
        return self._status

    @property
    async def exists(self) -> bool:
        print('exists starts')
        """Tell if the screen session exists or not."""
        # Parse the screen -ls call, to find if the screen exists or not.
        #  "	28062.G.Terminal	(Detached)"
        lines = (await _exec('screen -ls', connector=self._connector)).split('\n')
        # if self._connector is not None:
        #     lines = (await self._connector.execute('screen -ls')).stdout.split('\n')
        # else:
        #     lines = getoutput("screen -ls").split('\n')
        print('exists ends')
        return self.name in [".".join(l.split(".")[1:]).split("\t")[0]
                             for l in lines if self.name in l]


    async def enable_logs(self, filename=None):
        if filename is None:
            filename = self.name
        await self._screen_commands(f"logfile {filename}", "log on")
        self._logfilename = filename
        open(filename, 'w+')
        self.logs = tailf(filename)

    async def disable_logs(self, remove_logfile=False):
        await self._screen_commands("log off")
        if remove_logfile:
            await _exec(f'rm {self._logfilename}', connector=self._connector)
        self.logs = None

    async def interrupt(self):
        """Insert CTRL+C in the screen session"""
        await self._screen_commands("eval \"stuff \\003\"")

    async def kill(self):
        """Kill the screen applications then close the screen"""
        await self._screen_commands('quit')

    async def detach(self):
        """detach the screen"""
        print('detach starts')
        await self._check_exists()
        ID = await self.id
        await _exec(f"screen -d  {ID}", connector=self._connector)
        print('detach ends')

    async def send_commands(self, *commands, cwd:str='.'):
        """send commands to the active gnu-screen"""
        print('send_commands starts')
        await self._check_exists()
        for command in commands:
            await self._screen_commands(f'stuff " {command} " ',
                                        'eval "stuff \\015"', cwd=cwd)
            print(f'Command: {command}  sent')
            print('send_commands ends')

    async def add_user_access(self, unix_user_name):
        """allow to share your session with an other unix user"""
        await self._screen_commands('multiuser on', 'acladd ' + unix_user_name)

    async def _screen_commands(self, *commands, cwd:str='.'):
        """allow to insert generic screen specific commands
        a glossary of the existing screen command in `man screen`"""
        print('_screen_commands starts')
        await self._check_exists()
        await _exec(f'cd {cwd}', connector=self._connector)
        for command in commands:
            ID = await self.id
            await _exec(f'screen -dm -x {ID} -p 0 -X  {command}', connector=self._connector)
            # system('screen -x ' + await self.id + ' -X ' + command)
            await asyncio.sleep(0.02)
        print('_screen_commands ends')

    async def _check_exists(self, message="Error code: 404."):
        """check whereas the screen exist. if not, raise an exception"""
        if not (await self.exists):
            raise ScreenNotFoundError(message, self.name)

    async def _set_screen_infos(self):
        """set the screen information related parameters"""
        print('_set_screen_infos starts')
        # await self.lock.acquire()
        if (await self.exists):
            line = ""
            lines = (await _exec("screen -ls", connector=self._connector)).split("\n")
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
            self._id = int(infos[0].split('.')[0])
            if len(infos) == 3:
                self._date = infos[1][1:-1]
                self._status = infos[2][1:-1]
            else:
                self._status = infos[1][1:-1]
        print('_set_screen_infos ends')
        # await self.lock.release()

    async def _delayed_detach(self):
        await asyncio.sleep(0.5)
        # sleep(2)
        await self.detach()

    def __repr__(self):
        return "<%s '%s'>" % (self.__class__.__name__, self.name)
