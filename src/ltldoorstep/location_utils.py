import sys
import os
import json
import pandas
import berlin
import berlin.code_type
import logging
import re
from itertools import chain

DIR = os.path.dirname(__file__)

_berlin_interface = None

class BerlinInterface:
    data_sources = {
        'iata_file': 'locode/airport-codes/data/airport-codes.csv',
        'state_file': 'locode/country-codes/data/country-codes.csv',
        'subdiv_file': [
            'locode/un-locode/data/subdivision-codes.csv',
            'locode/iso-3166-2/iso_3166_2.js',
        ],
        'locode_file': 'locode/un-locode/data/code-list.csv'
    }

    def get_code(self, full_code):
        """
        Get Berlin Code for a code including its code type, in the format:
          CODETYPE#ST:LCD
        or similar.
        """

        return self._code_bank.from_identifier(full_code)

    def load(self):
        def full_path(f):
            if type(f) is list:
                return [full_path(i) for i in f]
            return os.path.join(DIR, '..', '..', 'tests', 'examples', 'data', f)

        local_sources = {k: full_path(v) for k, v in self.data_sources.items()}

        self._code_bank = berlin.code_type.get_code_bank(local_sources, progress_bar=False)

def load_berlin():
    global _berlin_interface

    if not _berlin_interface:
        _berlin_interface = BerlinInterface()
        _berlin_interface.load()

    return _berlin_interface
