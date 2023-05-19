"""
USPEX.Stages.Connector
======================

.. codeauthor:: Artem Samtsevich <samtsevichartem@gmail.com>

"""

import logging
import asyncio, asyncssh

from copy import copy
import random
from pathlib import Path

logger = logging.getLogger(__name__)
asyncssh.set_log_level(logging.WARNING)

N_TRIES = 100


class SSHConnectorClient(asyncssh.SSHClient):

    def __init__(self, callbackIfLost):
        self._conn = None
        self._callbackIfLost = callbackIfLost

    def connection_made(self, connection):
        self._conn = connection
        self._conn.set_keepalive(30)

    def connection_lost(self, exc):
        self._conn = None
        logger.debug('Connection lost')
        if self._callbackIfLost is not None:
            asyncio.ensure_future(self._callbackIfLost())

    def isValid(self):
        return self._conn is not None

    def __del__(self):
        if self.isValid():
            self._callbackIfLost = None
            self._conn.close()


class Connector(object):
    '''
    Class that is a bridge between computer with USPEX and supercomputer for a ab-initio calculations.
    '''

    MAX_CHAN = 10

    def __init__(self, domain: str = None, remoteFolder: str = '~/USPEXRemoteFolder', maxChannels=MAX_CHAN, **kwargs):
        '''
        :type domain: str
        :param domain: domain name or IP address of remote server.
        :type username: str
        :param username: username for the connection.
        :type client_keys: list of strings
        :param client_keys: list of paths to files containing ssh keys to be used with the connection. (optional)
        :type password: str
        :param password: password for the connection. (optional)
        :type know_hosts: str or NoneType
        :param: path to file containing list of known hosts. None if verification of known hosts is not needed.
        :type remoteFolder: Path
        :param remoteFolder: path to working folder on remote server.
        '''

        self._domain = domain
        self.remoteFolder = None if domain is None else Path(remoteFolder)
        self._maxChannels = maxChannels
        self._kwargs = kwargs

        self.conn = None
        self.client = None
        self._lock = asyncio.Lock()
        self.channelGuard = asyncio.Semaphore(self._maxChannels)

    ############################################
    # Code responds for serialization
    def __getstate__(self):
        state = copy(self.__dict__)
        del state['_lock']
        del state['channelGuard']
        del state['conn']
        del state['client']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.conn = None
        self.client = None
        self._lock = asyncio.Lock()
        self.channelGuard = asyncio.Semaphore(self._maxChannels)
    ############################################

    async def execute(self, execCommand: str, cwd: str = '.', **kwargs):
        '''
        Method for running some commands on remote server
        :param execCommand:
        :return: exitcode, error, output
        '''
        if self._domain is not None:
            sftp = await self._start_sftp_session()
            remote_cwd = str(self.remoteFolder/cwd).replace('~', await sftp.getcwd())
            await sftp.makedirs(remote_cwd, exist_ok=True)
            await self._close_sftp_session(sftp)
            command = f"cd {remote_cwd} && {execCommand}"
            remote_result = await self._run(command, **kwargs)
            return remote_result.returncode, remote_result.stdout, remote_result.stderr
        else:
            if 'stdin' not in kwargs:
                kwargs['stdin'] = asyncio.subprocess.PIPE
            if 'stdout' not in kwargs:
                kwargs['stdout'] = asyncio.subprocess.PIPE
            if 'stderr' not in kwargs:
                kwargs['stderr'] = asyncio.subprocess.PIPE
            if 'input' in kwargs:
                input = kwargs['input'].encode()
                kwargs['stdin'] = asyncio.subprocess.PIPE
                del kwargs['input']
            else:
                input = None
            process = await asyncio.create_subprocess_shell(execCommand, cwd=cwd, **kwargs)
            out, err = await process.communicate(input)
            if isinstance(out, bytes):
                out = out.decode()
            elif out is None:
                out = ''
            if isinstance(err, bytes):
                err = err.decode()
            elif err is None:
                err = ''
            returncode = process.returncode
            return returncode, out, err

    async def sync_l2r(self, path: Path):
        '''
        Local to remote
        :param path: path to directory
        '''
        if self._domain is not None:
            sftp = await self._start_sftp_session()
            remote_path = Path(str(self.remoteFolder/path).replace('~', await sftp.getcwd()))
            await sftp.makedirs(remote_path.parent, exist_ok=True)
            await sftp.put(path, remote_path, recurse=True)
            await self._close_sftp_session(sftp)

    async def sync_r2l(self, path: Path):
        '''
        Remote to local
        :param path: path to directory
        '''
        if self._domain is not None:
            sftp = await self._start_sftp_session()
            remote_path = Path(str(self.remoteFolder/path).replace('~', await sftp.getcwd()))
            local_path = path.parent if path.is_dir() else path
            await sftp.get(remote_path, local_path, recurse=True)
            await self._close_sftp_session(sftp)

    async def clean(self, path: Path):
        '''
        :param path: path to directory
        '''
        if self._domain is not None:
            sftp = await self._start_sftp_session()
            path = str(self.remoteFolder/path).replace('~', await sftp.getcwd())
            await sftp.rmtree(path, ignore_errors=True)
            await self._close_sftp_session(sftp)

    async def _checkConnection(self):
        await self._lock.acquire()
        if self.client is None or not self.client.isValid():
            self.conn, self.client = await asyncssh.create_connection(lambda: SSHConnectorClient(self._checkConnection),
                                                                      self._domain, **self._kwargs)
        self._lock.release()

    async def _run(self, *args, **kwargs):
        await self.channelGuard.acquire()
        for i in range(N_TRIES):
            await self._checkConnection()
            try:
                remote_result = await self.conn.run(*args, **kwargs)
                logger.debug(f"_run success ({i + 1}/{N_TRIES})")
            except Exception as e:
                logger.debug(e)
                pause = round(10 * (1 + random.random()))
                logger.debug(f"Retrying _run in {pause} seconds ({i + 1}/{N_TRIES})")
                await asyncio.sleep(pause)
                continue
            break
        else:
            self.channelGuard.release()
            raise RuntimeError('Cant run ssh command')
        self.channelGuard.release()
        return remote_result

    async def _start_sftp_session(self):
        await self.channelGuard.acquire()
        for i in range(N_TRIES):
            await self._checkConnection()
            try:
                sftp = await self.conn.start_sftp_client()
                logger.debug(f"_start_sftp_session success ({i + 1}/{N_TRIES})")
            except Exception as e:
                logger.debug(e)
                pause = round(10 * (1 + random.random()))
                logger.debug(f"Retrying _start_sftp_session in {pause} seconds ({i + 1}/{N_TRIES})")
                await asyncio.sleep(pause)
                continue
            break
        else:
            self.channelGuard.release()
            raise RuntimeError('Cant start sftp session')
        return sftp

    async def _close_sftp_session(self, sftp):
        assert sftp is not None
        sftp.exit()
        await sftp.wait_closed()
        self.channelGuard.release()
