'''
@file        Connector_Test.py
@author:     Pavel Bushlanov
@copyright:  2019 Oganov's Lab. All rights reserved.
@contact:    paulbusj@mail.ru
@date        1 April 2019
@brief       Class for testing Connector.
'''

import unittest
import os
import asyncio
import shutil

from USPEX.Calculators.Connector import Connector


class Connector_Test(unittest.TestCase):
    '''

    '''

    def test_sync_r2l(self):
        connector = Connector(domain = 'localhost', known_hosts=None)
        folder = 'TestFolder/TestFolder'
        remote_path = os.path.join(connector.remoteFolder, folder).replace('~', os.getenv('HOME'))
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder)
        shutil.rmtree(remote_path, ignore_errors=True)
        os.makedirs(remote_path)
        with open(os.path.join(remote_path, 'File'), 'wt') as f:
            f.write('Content')
        loop = asyncio.get_event_loop()
        loop.run_until_complete(connector.sync_r2l(folder))
        self.assertTrue(os.path.exists(os.path.join(folder, 'File')))
        with open(os.path.join(folder, 'File'), 'rt') as f:
            content = f.read()
        self.assertEqual(content, 'Content')
        shutil.rmtree(os.path.dirname(folder), ignore_errors=True)
        shutil.rmtree(os.path.dirname(remote_path), ignore_errors=True)

    def test_sync_l2r_and_clean(self):
        connector = Connector(domain = 'localhost', known_hosts=None)
        folder = 'TestFolder/TestFolder'
        remote_path = os.path.join(connector.remoteFolder, folder).replace('~', os.getenv('HOME'))
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder)
        with open(os.path.join(folder, 'File'), 'wt') as f:
            f.write('Content')
        loop = asyncio.get_event_loop()
        loop.run_until_complete(connector.sync_l2r(folder))
        self.assertTrue(os.path.exists(os.path.join(remote_path, 'File')))
        with open(os.path.join(remote_path, 'File'), 'rt') as f:
            content = f.read()
        self.assertEqual(content, 'Content')
        loop.run_until_complete(connector.clean(folder))
        self.assertFalse(os.path.exists(remote_path))
        self.assertTrue(os.path.exists(os.path.dirname(remote_path)))
        loop.run_until_complete(connector.clean(os.path.dirname(folder)))
        self.assertFalse(os.path.exists(os.path.dirname(remote_path)))
        shutil.rmtree(os.path.dirname(folder), ignore_errors=True)

    def test_execute_remote(self):
        connector = Connector(domain = 'localhost', known_hosts=None)
        loop = asyncio.get_event_loop()
        ret, out, err = loop.run_until_complete(connector.execute('pwd'))
        self.assertEqual(ret, 0)
        self.assertEqual(out.replace('\n', ''), connector.remoteFolder.replace('~', os.getenv('HOME')))
        self.assertEqual(err, '')
        ret, out, err = loop.run_until_complete(connector.execute('cat', input='Hi'))
        self.assertEqual(ret, 0)
        self.assertEqual(out.replace('\n', ''), 'Hi')
        self.assertEqual(err, '')
        shutil.rmtree('TestFolder', ignore_errors=True)
        os.makedirs('TestFolder')
        with open('TestFolder/input', 'wt') as f:
            f.write('Content')
        with open('TestFolder/input', 'r') as fi, open('TestFolder/output', 'w') as fo, open('TestFolder/error', 'w') as fe:
            ret, out, err = loop.run_until_complete(connector.execute('cat', stdin=fi, stdout=fo, stderr=fe))
        self.assertEqual(ret, 0)
        self.assertEqual(out, '')
        self.assertEqual(err, '')
        with open('TestFolder/output', 'r') as fo, open('TestFolder/error', 'r') as fe:
            self.assertEqual(fo.read(), 'Content')
            self.assertEqual(fe.read(), '')
        shutil.rmtree('TestFolder')

    def test_execute_local(self):
        connector = Connector()
        loop = asyncio.get_event_loop()
        ret, out, err = loop.run_until_complete(connector.execute('pwd'))
        self.assertEqual(ret, 0)
        self.assertEqual(out.replace('\n', ''), os.getcwd())
        self.assertEqual(err, '')
        ret, out, err = loop.run_until_complete(connector.execute('cat', input='Hi'))
        self.assertEqual(ret, 0)
        self.assertEqual(out.replace('\n', ''), 'Hi')
        self.assertEqual(err, '')
        shutil.rmtree('TestFolder', ignore_errors=True)
        os.makedirs('TestFolder')
        with open('TestFolder/input', 'wt') as f:
            f.write('Content')
        with open('TestFolder/input', 'r') as fi, open('TestFolder/output', 'w') as fo, open('TestFolder/error', 'w') as fe:
            ret, out, err = loop.run_until_complete(connector.execute('cat', stdin=fi, stdout=fo, stderr=fe))
        self.assertEqual(ret, 0)
        self.assertEqual(out, '')
        self.assertEqual(err, '')
        with open('TestFolder/output', 'r') as fo, open('TestFolder/error', 'r') as fe:
            self.assertEqual(fo.read(), 'Content')
            self.assertEqual(fe.read(), '')
        shutil.rmtree('TestFolder')

