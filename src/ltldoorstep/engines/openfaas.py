"""Engine for running a job on an OpenFaaS cluster."""

from contextlib import contextmanager
import requests
from requests.auth import HTTPBasicAuth
import asyncio
import time
import os
import sys
import json
import uuid
import jinja2
import logging
from concurrent.futures import ThreadPoolExecutor
from .engine import Engine
from ..reports.tabular import TabularReport

ALLOWED_IMAGES = [
    ('lintol/doorstep', 'latest'),
    ('lintol/ds-csvlint', 'latest')
]


class OpenFaaSEngine(Engine):
    """Allow execution of workflows on a OpenFaaS cluster."""

    def __init__(self, config=None):
        pass

    @staticmethod
    def description():
        return '(not provided)'

    @staticmethod
    def config_help():
        return None

    def add_data(self, filename, content, redirect, session):
        raise NotImplementedError("Function must be implemented")

    def add_processor(self, modules, metadata, session):
        with open('/home/philtweir/Code/Projects/Lintol/openfaas-w', 'r') as f:
            pw = f.read()
        rq = requests.post('http://192.168.39.150:31112/system/functions', data={
            "service": "stronghash2",
            "image": "functions/alpine",
            "envProcess": "sha512sum",
            "network": "func_functions"
        }, auth=HTTPBasicAuth('admin', pw))

        print(rq)

    async def run(self, filename, workflow_module, metadata, bucket=None):
        return await self._run()

    async def _run(self):
        with open('/tmp/openfaas', 'r') as f:
            pw = f.read()
        rq = requests.get('http://192.168.39.150:31112/system/functions', data={
            "service": "stronghash3",
            "image": "functions/alpine",
            "envProcess": "sha512sum",
            "network": "func_functions"
        }, auth=HTTPBasicAuth('admin', pw))

        rq = requests.post('http://192.168.39.150:31112/function/stronghash3', data={
        }, auth=HTTPBasicAuth('admin', pw))
        print(rq.content)

        report = TabularReport('dummy-openfaas-test', 'Dummy OpenFaaS Test', filename='/tmp/dummy.csv')
        report.add_issue(logging.WARN, 'test-openfaas', rq.content)

        return report.compile()

    async def monitor_pipeline(self, session):
        raise NotImplementedError("Function must be implemented")

    async def get_output(self, session):
        raise NotImplementedError("Function must be implemented")

    @contextmanager
    def make_session(self):
        raise NotImplementedError("Function must be implemented")
