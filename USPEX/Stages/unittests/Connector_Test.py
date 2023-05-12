'''
@file        Connector_Test.py
@author:     Pavel Bushlanov
@copyright:  2019 Oganov's Lab. All rights reserved.
@contact:    paulbusj@mail.ru
@date        1 April 2019
@brief       Class for testing Connector.
'''

import unittest
import asyncio
import shutil

from pathlib import Path

from ..Connector import Connector


class Connector_Test(unittest.TestCase):
    '''

    '''

    def test_sync_r2l(self):
        connector = Connector(domain = 'localhost', known_hosts=None)
        folder = Path('TestFolder/TestFolder')
        remote_path = Path(str(connector.remoteFolder/folder).replace('~', str(Path.home())))
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        shutil.rmtree(remote_path, ignore_errors=True)
        remote_path.mkdir(parents=True)
        with open(remote_path/'File', 'wt') as f:
            f.write('Content')
        loop = asyncio.get_event_loop()
        loop.run_until_complete(connector.sync_r2l(folder))
        self.assertTrue(folder.joinpath('File').exists())
        with open(folder/'File', 'rt') as f:
            content = f.read()
        self.assertEqual(content, 'Content')
        shutil.rmtree(folder.parent, ignore_errors=True)
        shutil.rmtree(remote_path.parent, ignore_errors=True)

    def test_sync_l2r_and_clean(self):
        connector = Connector(domain = 'localhost', known_hosts=None)
        folder = Path('TestFolder/TestFolder')
        remote_path = Path(str(connector.remoteFolder/folder).replace('~', str(Path.home())))
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        with open(folder/'File', 'wt') as f:
            f.write('Content')
        loop = asyncio.get_event_loop()
        loop.run_until_complete(connector.sync_l2r(folder))
        self.assertTrue(remote_path.joinpath('File').exists())
        with open(remote_path/'File', 'rt') as f:
            content = f.read()
        self.assertEqual(content, 'Content')
        loop.run_until_complete(connector.clean(folder))
        self.assertFalse(remote_path.exists())
        self.assertTrue(remote_path.parent.exists())
        loop.run_until_complete(connector.clean(folder.parent))
        self.assertFalse(remote_path.parent.exists())
        shutil.rmtree(folder.parent, ignore_errors=True)

    def test_execute_remote(self):
        connector = Connector(domain = 'localhost', known_hosts=None)
        loop = asyncio.get_event_loop()
        ret, out, err = loop.run_until_complete(connector.execute('pwd'))
        self.assertEqual(ret, 0)
        self.assertEqual(out.replace('\n', ''), str(connector.remoteFolder).replace('~', str(Path.home())))
        self.assertEqual(err, '')
        ret, out, err = loop.run_until_complete(connector.execute('cat', input='Hi'))
        self.assertEqual(ret, 0)
        self.assertEqual(out.replace('\n', ''), 'Hi')
        self.assertEqual(err, '')
        test_fold = Path('TestFolder')
        shutil.rmtree(test_fold, ignore_errors=True)
        test_fold.mkdir(parents=True)
        with open(test_fold/'input', 'wt') as f:
            f.write('Content')
        with open(test_fold/'input', 'r') as fi, open(test_fold/'output', 'w') as fo, open(test_fold/'error', 'w') as fe:
            ret, out, err = loop.run_until_complete(connector.execute('cat', stdin=fi, stdout=fo, stderr=fe))
        self.assertEqual(ret, 0)
        self.assertEqual(out, '')
        self.assertEqual(err, '')
        with open(test_fold/'output', 'r') as fo, open(test_fold/'error', 'r') as fe:
            self.assertEqual(fo.read(), 'Content')
            self.assertEqual(fe.read(), '')
        shutil.rmtree(test_fold)

    def test_execute_local(self):
        connector = Connector()
        loop = asyncio.get_event_loop()
        ret, out, err = loop.run_until_complete(connector.execute('pwd'))
        self.assertEqual(ret, 0)
        self.assertEqual(out.replace('\n', ''), str(Path.cwd()))
        self.assertEqual(err, '')
        ret, out, err = loop.run_until_complete(connector.execute('cat', input='Hi'))
        self.assertEqual(ret, 0)
        self.assertEqual(out.replace('\n', ''), 'Hi')
        self.assertEqual(err, '')
        test_fold = Path('TestFolder')
        shutil.rmtree(test_fold, ignore_errors=True)
        test_fold.mkdir(parents=True)
        with open(test_fold/'input', 'wt') as f:
            f.write('Content')
        with open(test_fold/'input', 'r') as fi, open(test_fold/'output', 'w') as fo, open(test_fold/'error', 'w') as fe:
            ret, out, err = loop.run_until_complete(connector.execute('cat', stdin=fi, stdout=fo, stderr=fe))
        self.assertEqual(ret, 0)
        self.assertEqual(out, '')
        self.assertEqual(err, '')
        with open(test_fold/'output', 'r') as fo, open(test_fold/'error', 'r') as fe:
            self.assertEqual(fo.read(), 'Content')
            self.assertEqual(fe.read(), '')
        shutil.rmtree(test_fold)
