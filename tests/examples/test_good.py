import os
from dask.threaded import get
import gettext

if __name__ == "__main__":
    gettext.install('ltldoorstep')

from ltldoorstep_examples.good import processor

def test_good_on_bad():
    path = os.path.join(os.path.dirname(__file__), 'data', 'awful.csv')
    good = processor()
    workflow = good.get_workflow(path)
    results = get(workflow, 'output').compile()

    errors = results['tables'][0]['errors']
    print(errors)
    assert len(errors) == 7

    report = errors[0]
    print(report)
    assert report['code'] == 'duplicate-header'

if __name__ == "__main__":
    test_good_on_bad()
