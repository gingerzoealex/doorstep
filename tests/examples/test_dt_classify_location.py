import os
from dask.threaded import get
from ltldoorstep_examples.dt_classify_location import processor
from ltldoorstep.metadata import DoorstepMetadata
import pytest
import json

def testing_dt_classify_location_pass():
    path = os.path.join(os.path.dirname(__file__), 'data', 'number-of-licensed-bus-vehicles-by-postal-district-at-30-september-2018.csv')
    metadata_path = os.path.join(os.path.dirname(__file__), 'sample', 'metadata-buses.json')
    classify_location = processor()

    with open(metadata_path, 'r') as metadata_f:
        metadata = json.load(metadata_f)

    workflow = classify_location.build_workflow(path, metadata)
    report = get(workflow, 'output')
    results = report.compile()

    print(json.dumps(report.compile(), indent=True))

    assert type(classify_location.metadata) is DoorstepMetadata
    assert type(classify_location.metadata.package) is dict
