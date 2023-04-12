"""
USPEX.Stages.Connector
======================

.. codeauthor:: Artem Samtsevich <samtsevichartem@gmail.com>

"""

import logging
import asyncio, asyncssh
import os
from copy import copy

logger = logging.getLogger(__name__)
asyncssh.set_log_level(logging.WARNING)


class SSHConnectorClient(asyncssh.SSHClient):

    def __init__(self, callbackIfLost):
        self._conn = None
        self._callbackIfLost = callbackIfLost

    def connection_made(self, connection):
        self._conn = connection

    def connection_lost(self, exc):
        self._conn = None
        logger.debug('Connection lost')
        if self._callbackIfLost is not None:
            self._callbackIfLost()

    def isValid(self):
        return self._conn is not None

    def __del__(self):
        if self.isValid():
            self._callbackIfLost = None
            self._conn.close()


class SFTPSession:
    def __init__(self, conn, guard):
        self._conn = conn
        self._guard = guard
        self._sftp = None

    async def start(self):
        assert self._sftp is None
        await self._guard.acquire()
        self._sftp = await self._conn.start_sftp_client()

    async def close(self):
        assert self._sftp is not None
        self._sftp.exit()
        await self._sftp.wait_closed()
        self._guard.release()
        self._sftp = None

    async def remove(self, path : str):
        '''
        :param path: path to directory
        '''
        if await self._sftp.exists(path):
            if await self._sftp.isdir(path):
                for name in await self.listdir(path):
                    await self.remove(os.path.join(path,name))
                await self._sftp.rmdir(path)
            else:
                await self._sftp.remove(path)

    async def makedirs(self, path : str):
        '''
        :param path: path: path to directory
        :return:
        '''
        if not await self._sftp.isdir(path):
            try:
                if await self._sftp.exists(path):
                    await self._sftp.remove(path)
                await self._sftp.mkdir(path)
            except asyncssh.sftp.SFTPError:
                head, tail = os.path.split(path)
                await self.makedirs(head)
                if tail != '.':
                    await self._sftp.mkdir(path)

    async def listdir(self, path : str) -> list:
        '''
        :param path: path to directory
        '''
        res = []
        for name in await self._sftp.listdir(path):
            if name != '.' and name != '..':
                res.append(name)
        return res

    def __getattr__(self, item):
        if item == '__setstate__':
            raise AttributeError
        if hasattr(self._sftp, item):
            return getattr(self._sftp, item)
        else:
            raise AttributeError



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
        :type remoteFolder: str
        :param remoteFolder: path to working folder on remote server.
        '''

        self._domain = domain
        if domain is None:
            self.remoteFolder = None
        else:
            self.remoteFolder = remoteFolder
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

    async def execute(self, execCommand : str, cwd : str = '.', **kwargs):
        '''
        Method for running some commands on remote server
        :param execCommand:
        :return: exitcode, error, output
        '''
        if self._domain is not None:
            sftp = await self._start_sftp_session()
            remote_cwd = os.path.join(self.remoteFolder, cwd).replace('~', await sftp.getcwd())
            await sftp.makedirs(remote_cwd)
            await sftp.close()
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

    async def sync_l2r(self, path : str):
        '''
        Local to remote
        :param path: path to directory
        '''
        if self._domain is not None:
            sftp = await self._start_sftp_session()
            remote_path = os.path.join(self.remoteFolder, path).replace('~', await sftp.getcwd())
            await sftp.remove(remote_path)
            await sftp.makedirs(os.path.dirname(remote_path))
            await sftp.put(path, remote_path, recurse = True)
            await sftp.close()

    async def sync_r2l(self, path : str):
        '''
        Remote to local
        :param path: path to directory
        '''
        if self._domain is not None:
            sftp = await self._start_sftp_session()
            remote_path = os.path.join(self.remoteFolder, path).replace('~', await sftp.getcwd())
            if not await sftp.isdir(remote_path):
                await sftp.get(remote_path, path)
            else:
                filenames = await sftp.listdir(remote_path)
                for filename in filenames:
                    await sftp.get(os.path.join(remote_path, filename), os.path.join(path, filename), recurse = True)
            await sftp.close()

    async def clean(self, path : str):
        '''
        :param path: path to directory
        '''
        if self._domain is not None:
            sftp = await self._start_sftp_session()
            path = os.path.join(self.remoteFolder, path).replace('~', await sftp.getcwd())
            await sftp.remove(path)
            await sftp.close()

    def _makeConnection(self):
        self.conn, self.client = await asyncssh.create_connection(lambda: SSHConnectorClient(self._makeConnection),
                                                                  self._domain, **self._kwargs)

    async def _checkConnection(self):
        await self._lock.acquire()
        doCheck = True
        while doCheck:
            if self.client is None or not self.client.isValid():
                self._makeConnection()
            else:
                try:
                    await self.channelGuard.acquire()
                    sftp = await self.conn.start_sftp_client()
                    await sftp.getcwd()
                    sftp.exit()
                    await sftp.wait_closed()
                    self.channelGuard.release()
                except Exception:
                    logger.exception("Exception in checkConnection.")
                    continue
            doCheck = False
        self._lock.release()

    async def _run(self, *args, **kwargs):
        await self._checkConnection()
        await self.channelGuard.acquire()
        remote_result = await self.conn.run(*args, **kwargs)
        self.channelGuard.release()
        return remote_result

    async def _start_sftp_session(self):
        await self._checkConnection()
        sftp = SFTPSession(self.conn, self.channelGuard)
        await sftp.start()
        return sftp
