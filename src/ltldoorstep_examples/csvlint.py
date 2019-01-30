from goodtables import validate
import logging
import os
import sys
import pandas as p
from dask.threaded import get
from ltldoorstep.processor import DoorstepProcessor
from ltldoorstep.sandbox import run_in_sandbox

CSVLINT_IMAGE = 'lintol/doorstep'

def sandbox_ruby(filename):
    return run_in_sandbox(CSVLINT_IMAGE, ['ltldoorstep', 'process', '/input.file', 'examples/processors/good.py'], filename, '/input.file')

class CsvLintProcessor(DoorstepProcessor):
    def get_workflow(self, filename, metadata={}):
        workflow = {
            'output': (sandbox_ruby, filename)
        }
        return workflow

processor = CsvLintProcessor

if __name__ == "__main__":
    argv = sys.argv
    processor = CsvLintProcessor()
    workflow = processor.build_workflow(argv[1])
