"""This script will attempt to validate any json register files from gov.uk

This will validate all government names e.g. council, country, street names
The first function will check to see if the json file provided to the function contains certain features
If they are contained, it is valid. If not, the json is not a valid register
Second function checks if ids are unique
Third function checks if ids are surjective i.e. are any ids missing?
Fourth function creates a dict object that loads the json file and runs each function and returns it.

"""

import json
import gettext
import pandas as p
import logging
import sys
import os
from fuzzywuzzy import process, fuzz
import ltldoorstep
from ltldoorstep.processor import DoorstepProcessor
from ltldoorstep.reports import report
from dask.threaded import get


EXCLUDE_EMPTY_MATCHES = True
PROVIDE_SUGGESTIONS = True
GUESS_THRESHOLD = 85.0
DEFAULT_REGISTER_LOCATION = os.path.join(os.path.dirname(ltldoorstep.__file__), '../..', 'tests', 'examples', 'data', 'register-countries.json')


def set_properties(df, rprt):
    rprt.set_properties(headers=list(df.columns))
    return rprt


class RegisterCountryItem:
    matches = ()
    heading = None

    def __init__(self, heading, matches, suggest=False, preprocess=None):
        self.heading = heading
        self.matches = matches
        self.suggest = suggest and PROVIDE_SUGGESTIONS
        self.preprocess = preprocess

    def load_allowed(self, countries):
        rows = [country[self.heading] for country in countries.values()]
        self.allowed = set()

        for row in rows:
            if self.preprocess:
                row = self.preprocess(row)

            if isinstance(row, list):
                self.allowed.update(row)
            else:
                self.allowed.add(row)

    def find_columns(self, data):
        columns = self.matches & {c.lower() for c in data.columns}
        return columns

    def analyse(self, value):
        analysis = {'mismatch': value}

        if self.suggest:
            analysis['guess'] = tuple(process.extractOne(value, self.allowed, scorer=fuzz.token_set_ratio))

        return analysis

    def mismatch(self, series):
        return ~series.isin(self.allowed)


"""This is a dict object that contains categories commonly contained within a gov.uk register
 Data passed in will be checked against this in gov_register_checker
 This could be changed to suit different needs. """

possible_columns = [
    RegisterCountryItem(
        'country',
        {'countrycode', 'country', 'state', 'nationality'}
    ),
    RegisterCountryItem(
        'official-name',
        {'country', 'country_name'},
        suggest=True
    ),
    RegisterCountryItem(
        'name',
        {'state', 'country', 'country_name'},
        suggest=True
    ),
    RegisterCountryItem(
        'citizen-names',
        {'nationality'},
        suggest=True,
        preprocess=lambda r: [entry.replace('citizen', '').strip() for entry in r.split(';')]
    )
]

def find_relevant_columns(data):
    columns = {k: [] for k in data.columns}

    for col in possible_columns:
        for k in col.find_columns(data):
            columns[k].append(col)

    return {k: v for k, v in columns.items() if v}

def best_matching_series(series, columns):
    series = series.dropna()
    matches = [(col, series[col.mismatch(series)]) for col in columns]

    if EXCLUDE_EMPTY_MATCHES:
        matches = [match for match in matches if len(match[1]) < len(series)]

    if matches:
        col, series = min(matches, key=lambda m: len(m[1]))
        return col.heading, {k: col.analyse(v) for k, v in series.iteritems()}

    return None

def gov_countries_register_checker(data, rprt):
    # making test_data into data loaded from json
    register_filename = os.path.join(DEFAULT_REGISTER_LOCATION)

    with open(register_filename, 'r') as data_file:
        countries = {row['key']: row['item'][0] for row in json.load(data_file).values()}

    for col in possible_columns:
        col.load_allowed(countries)

    columns = find_relevant_columns(data)

    issues = {k: best_matching_series(data[k], v) for k, v in columns.items()}

    matrix = {k: issue for  k, issue in issues.items() if issue}

    mismatching_columns = {k: issue[1] for k, issue in matrix.items() if issue[1]}
    for column, issues in mismatching_columns.items():
        for row, issue in issues.items():
            issue_text = _("Unexpected country-related term '%s'") % issue['mismatch']
            if 'guess' in issue and issue['guess'][1] > GUESS_THRESHOLD:
                issue_text += _(", perhaps you meant '%s'?") % issue['guess'][0]

            column_number = data.columns.get_loc(column)
            row_number = data.index.get_loc(row)
            native_row = json.loads(data.loc[row].to_json())
            rprt.add_issue(
                logging.WARNING,
                'country-mismatch',
                issue_text,
                column_number=column_number,
                row_number=row_number,
                row=native_row,
                error_data=issue
            )

    checked_columns = {k: 'country-register-{col}'.format(col=issue[0]) for k, issue in matrix.items()}
    for column, checker in checked_columns.items():
        _("%s using checker %s")
        rprt.add_issue(
            logging.INFO,
            'country-checked',
            _("Column %s was checked with state attribute checker %s") % (column, checker),
            column_number=data.columns.get_loc(column),
            error_data=checked_columns
        )

    return rprt

"""This is the workflow builder.

This function will feed the json file into each method and then return the result
"""

class RegisterCountryProcessor(DoorstepProcessor):
    @staticmethod
    def make_report():
        return report.TabularReport('lintol/gov-uk-register-countries:1', _("Gov UK Registers Processor"))

    def get_workflow(self, filename, metadata={}):
        # setting up workflow dict
        workflow = {
            'load_csv' : (p.read_csv, filename),
            'properties': (set_properties, 'load_csv', self._report),
            'output' : (gov_countries_register_checker, 'load_csv', 'properties')
        }
        return workflow

processor = RegisterCountryProcessor.make

if __name__ == '__main__':
    gettext.install('ltldoorstep')
    proc = processor()
    workflow = proc.build_workflow(sys.argv[1])
    print(get(workflow, 'output'))
