"""Check dask threaded engine is correctly functioning."""

from unittest.mock import Mock, patch, mock_open
import pytest
import asyncio
from ltldoorstep.engines.dask_threaded import DaskThreadedEngine
from ltldoorstep.processor import DoorstepProcessor

import logging

@pytest.fixture
def engine():
    """Create a threaded dask engine."""

    eng = DaskThreadedEngine()

    return eng

class TestProcessor(DoorstepProcessor):
    preset = 'tabular'
    code = 'testing-processor'

    def ret(self, r, filename, metadata):
        r.add_issue(logging.ERROR, 'foo-bar', filename.upper())

    def get_workflow(self, filename, metadata):
        return {'output': (self.ret, self._report, filename, metadata)}

def test_can_run_workflow(engine):
    """Check we can run a workflow on local files."""

    filename = '/tmp/foo.csv'
    module = '/tmp/bar.py'
    metadata = {}

    mopen = mock_open(read_data='{}')
    source_file_loader_path = 'ltldoorstep.engines.dask_threaded.SourceFileLoader'
    source_file_loader = Mock()

    module = Mock()

    source_file_loader().load_module.return_value = module
    module.processor = TestProcessor.make
    loop = asyncio.get_event_loop()

    with patch('ltldoorstep.engines.dask_threaded.open', mopen) as _, \
            patch(source_file_loader_path, source_file_loader):
        result = loop.run_until_complete(engine.run(filename, module, metadata))

    assert result['tables'][0]['errors'][0]['message'] == filename.upper()
