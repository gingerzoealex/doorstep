import os
from dask.threaded import get
import logging
import pytest
from collections import defaultdict

if __name__ == "__main__":
    gettext.install('ltldoorstep')

from ltldoorstep.reports import report
from ltldoorstep import processor
from ltldoorstep_examples.boundary_checker_impr import BoundaryCheckerImprovedProcessor


def test_boundary_checker_on_pedestrian_crossings():
    path = os.path.join(os.path.dirname(__file__), 'data', 'pedestriancrossing.geojson')
    boundary_checker = BoundaryCheckerImprovedProcessor()

    ni_data = os.path.join(os.path.dirname(__file__), 'data', 'osni-ni-outline-lowres.geojson')
    metadata = {
        'definition': {},
        'supplementary': {'test.geojson': {'location': ni_data, 'source': 'http://example.com/ni_data.geojson'}},
        'configuration': {'boundary': '$->test.geojson'}
    }

    workflow = boundary_checker.get_workflow(path, metadata)
    report = get(workflow, 'output')
    results = report.compile()
    assert results is not None
    assert type(results) is dict
    assert len(results) == 9

    report = results['tables'][0]['errors']
    assert report is not None
    assert type(report) is list
    assert len(report) == 2
    assert report[0]['item']['entity']['location']['index'] == 1191
    assert report[1]['item']['entity']['location']['index'] == 1297
