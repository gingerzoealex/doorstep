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
from nltk.tag.stanford import StanfordNERTagger
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize import word_tokenize
from ltldoorstep.reports.report import TabularReport, combine_reports
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


def check_regex(df, rprt, rx, code, error_message):
    print(_("Regex %s starting") % code)
    for rix, (__, row) in enumerate(df.iterrows()):
        for cix, (__, cell) in enumerate(row.iteritems()):
            if rx.match(str(cell)):
                rprt.add_issue(
                    logging.ERROR,
                    'check_regex:%s' % code,
                    error_message,
                    row_number=rix,
                    column_number=cix,
                    error_data=None
                )
        print(rix)
    print(_("Regex %s done") % code, flush=True)
    return rprt

def check_mac(df, rprt):
    rx = r'((\d|([a-f]|[A-F])){2}:){5}(\d|([a-f]|[A-F])){2}'
    return check_regex(df, rprt, re.compile(rx), 'mac-address', _("Possible MAC address found"))

def check_postcodes(df, rprt):
    rx = r'([Gg][Ii][Rr] 0[Aa]{2})|((([A-Za-z][0-9]{1,2})|(([A-Za-z][A-Ha-hJ-Yj-y][0-9]{1,2})|(([A-Za-z][0-9][A-Za-z])|([A-Za-z][A-Ha-hJ-Yj-y][0-9]?[A-Za-z])))) ?[0-9][A-Za-z]{2})'
    return check_regex(df, rprt, re.compile(rx), 'uk-postcode', _("Possible UK postcode found"))

def check_email(df, rprt):
    rx = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,63}'
    return check_regex(df, rprt, re.compile(rx), 'email', _("Possible email address found"))

def check_ips(df, rprt):
    rx = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    return check_regex(df, rprt, re.compile(rx), 'ip', _("Possible IP address found"))


def set_properties(df, rprt):
    rprt.set_properties(headers=list(df.columns))
    return rprt


def check_nltk(df, rprt):
    ner = StanfordNERTagger('classifiers/english.all.3class.distsim.crf.ser.gz')

    tokenizer = TreebankWordTokenizer()
    tokens = []
    cells = []

    print("%d rows" % len(df.index))
    for rix, row in df.iterrows():
        for cix, cell in row.iteritems():
            try:
                words = tokenizer.tokenize(cell)
            except TypeError:
                pass
            else:
                tokens.append(words)
                cells.append((rix, cix))
        print("%d rows complete" % rix, flush=True)

    tags = ner.tag_sents(tokens)

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
                logging.ERROR,
                code,
                _("One or more potential named entities found, tagged as %s") % ', '.join(analysis.keys()),
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
            'email': (check_email, 'read', self.make_report()),
            'mac': (check_mac, 'read', self.make_report()),
            'postcodes': (check_postcodes, 'read', self.make_report()),
            'regex': (workflow_condense, 'ips', 'email', 'mac', 'postcodes'),
            'nltk': (check_nltk, 'read', 'regex'),
            'properties': (set_properties, 'read', 'nltk'),
            'output': (lambda rprt: (None, rprt), 'properties')
        }

        # Returns workflow dict
        return workflow

def workflow_condense(base, *args):
    return combine_reports(*args, base=base)

processor = PiiProcessor.make

if __name__ == "__main__":
    argv = sys.argv
    processor = PiiProcessor()
    workflow = processor.build_workflow(argv[1])
    print(get(workflow, 'output'))
