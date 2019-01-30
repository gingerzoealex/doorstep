"""PII Checker

This function will return the results of a csv file to find
personally identifable info(PII)

"""

import re
import socket
from contextlib import contextmanager
import logging
import requests
import numpy as np
import pandas as p
from dask.threaded import get
import sys
from ltldoorstep.processor import DoorstepProcessor
from nltk.parse import CoreNLPParser
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize import word_tokenize
from ltldoorstep.reports.report import TabularReport
import subprocess
import time
import os

pii_details = {
    'N': 'name',
    'P': 'phone_no',
    'A': 'address',
    'C': 'credit_card_no',
    'E': 'email_address',
    'O': 'organization'
}

@contextmanager
def run_nlp_server():
    root = os.environ['STANFORD_MODELS']
    jar_prefixes = {'stanford-corenlp', 'slf4j'}
    class_path = ':'.join([
        os.path.join(root, jar)
        for jar in os.listdir(root)
        if jar.endswith('.jar')
    ])
    command = [
        'java',
        '-mx1024m',
        '-cp',
        '"%s"' % class_path,
        'edu.stanford.nlp.pipeline.StanfordCoreNLPServer',
        '-port',
        '9000',
        '-timeout',
        '10000',
        '-loadClassifier',
        'edu/stanford/nlp/models/ner/english.all.3class.distsim.crf.ser.gz'
    ]

    env = {k: v for k, v in os.environ.items() if k in ('CLASSPATH')}

    nlp_process = subprocess.Popen(command, env=env, stdout=sys.stdout, stderr=sys.stdout)

    s = socket.socket()
    s.settimeout(30)
    for __ in range(6):
        try:
            s.connect(('localhost', 9000))
        except socket.error:
            time.sleep(5)
        else:
            break
    s.close()

    yield nlp_process
    nlp_process.kill()


def check_regex(df, rprt, rx, code, error_message):
    for rix, (__, row) in enumerate(df.iterrows()):
        for cix, (__, cell) in enumerate(row.iteritems()):
            if rx.match(str(cell)):
                rprt.add_issue(
                    logging.INFO,
                    'check_regex:%s' % code,
                    _("Possible IP found"),
                    row_number=rix,
                    column_number=cix,
                    error_data=None
                )
    return rprt

def check_ips(df, rprt):
    rx = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    return check_regex(df, rprt, re.compile(rx), 'ip', _("Possible IP address found"))


def check_nltk(df, rprt):
    with run_nlp_server() as server_process:
        ner = CoreNLPParser()

        tokenizer = TreebankWordTokenizer()
        tokens = []
        cells = []

        for rix, row in df.iterrows():
            for cix, cell in row.iteritems():
                try:
                    words = tokenizer.tokenize(cell)
                except TypeError:
                    pass
                else:
                    tokens.append(words)
                    cells.append((rix, cix))

        for __ in range(5):
            try:
                tags = ner.tag_sents(tokens)
            except requests.exceptions.ReadTimeout:
                pass
            else:
                break

    for ix, pairs in zip(cells, tags):
        analysis = {}

        for word, family in pairs:
            if family != 'O':
                if family not in analysis:
                    analysis[family] = set()
                analysis[family].add(word)

        if analysis:
            code = 'check_pii_detail:ner-found'
            row_number = df.index.get_loc(ix[0])
            column_number = df.columns.get_loc(ix[1])
            rprt.add_issue(
                logging.INFO,
                code,
                _("Potential named entity found, tagged as %s") % ', '.join(analysis.keys()),
                row_number=row_number,
                column_number=column_number,
                error_data={
                    'words-tagged': sum([len(words) for words in analysis.values()]),
                    'ner-types': list(analysis.keys())
                }
            )

    return rprt


class PiiProcessor(DoorstepProcessor):
    preset = 'tabular'
    code = 'lintol-pii-checker'
    description = _("Info from PII processor")

    def get_workflow(self, filename, metadata={}):
        """Workflow builder

        This function will return the workflow taken from return_report,
        feeds a csv file into return_report

        """

        # Setting up workflow dict...
        #  loading: Using Pandas library to read csv file passed in,
        #           and also pass the filename
        #  report_returned: Using return_report and loading csv file....
        #  output: This will be the output
        workflow = {
            'read': (p.read_csv, filename),
            'ips': (check_ips, 'read', self.make_report()),
            'output': (check_nltk, 'read', 'ips')
        }

        # Returns workflow dict
        return workflow

processor = PiiProcessor.make

if __name__ == "__main__":
    argv = sys.argv
    processor = PiiProcessor()
    workflow = processor.build_workflow(argv[1])
    print(get(workflow, 'output'))
