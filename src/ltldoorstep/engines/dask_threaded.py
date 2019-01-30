"""Engine for running a job, using dask, within this process."""

import uuid
import asyncio
from contextlib import contextmanager
from importlib.machinery import SourceFileLoader
from ..file import make_file_manager
from ..reports.report import combine_reports
from .dask_common import run
from .engine import Engine
from ..errors import LintolDoorstepException


class DaskThreadedEngine(Engine):
    """Allow execution of a dask workflow within this process."""

    def add_data(self, filename, content, redirect, session):
        session['data-filename'] = filename
        session['data-content'] = content
        data = {
            'filename': filename,
            'content': content
        }
        asyncio.ensure_future(session['queue'].put(data))

    def add_processor(self, modules, metadata, session):
        session['metadata'] = metadata

        if 'lang' in metadata:
            # TODO: check lang is valid
            lang = metadata['lang']

        if 'processors' not in session:
            session['processors'] = []

        for uid, processor in metadata['definitions'].items():
            configuration = {
                'name': uid,
                'definition': processor['definition'] if 'definition' in processor else {},
                'configuration': processor['configuration'] if 'configuration' in processor else {},
                'settings': processor['settings'] if 'settings' in processor else {},
                'supplementary': processor['supplementary'] if 'supplementary' in processor else {}
            }

            filename = None
            content = None
            if 'module' in processor:
                filename = processor['module']
                if processor['module'] in modules:
                    content = modules[processor['module']]
                else:
                    error_msg = _("Module content missing from processor %s") % processor['module']
                    logging.error(error_msg)
                    raise RuntimeError(error_msg)

            session['processors'].append({
                'name' : uid,
                'filename': filename,
                'content': content,
                'metadata': configuration
            })

    async def get_output(self, session):
        await session['completion'].acquire()

        result = session['result']

        session['completion'].release()

        if isinstance(result, LintolDoorstepException):
            raise result

        return json_dumps(result)

    async def monitor_pipeline(self, session):
        session['completion'] = asyncio.Lock()

        async def run_when_ready():
            session['completion'].acquire()
            data = await session['queue'].get()

            metadata = session['metadata']

            try:
                for processor in session['processors']:
                    result = await self.run(
                        data['filename'],
                        processor['filename'],
                        processor['metadata']
                    )
                session['result'] = result
            except Exception as error:
                if not isinstance(error, LintolDoorstepException):
                    error.status_code = None
                    error = LintolDoorstepException(error)
                session['result'] = error
            finally:
                session['completion'].release()

        asyncio.ensure_future(run_when_ready())

        return (False, session['completion'].acquire())

    @staticmethod
    async def run(filename, workflow_module, metadata, bucket=None):
        """Start the multi-threaded execution process."""

        mod = SourceFileLoader('custom_processor', workflow_module)

        result = None
        with make_file_manager(bucket) as file_manager:
            local_file = file_manager.get(filename)
            result = run(local_file, mod.load_module(), metadata)

        return result

    @contextmanager
    def make_session(self):
        """Set up a workflow session.

        This creates a self-contained set of dask constructs representing our operation.
        """

        name = 'doorstep-%s' % str(uuid.uuid4())
        data_name = '%s-data' % name
        #processors_name = '%s-processors' % name

        session = {
            'name': name,
            'data': data_name,
            'processors': [],
            'queue': asyncio.Queue()
        }

        yield session
