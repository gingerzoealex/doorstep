"""This script will attempt to classify any files by semantic location

"""

import json
import gettext
import pandas as p
import logging
import sys
import os
import ltldoorstep
from ltldoorstep.processor import DoorstepProcessor
from ltldoorstep.reports import report
from ltldoorstep.location_utils import load_berlin
from dask.threaded import get


EXCLUDE_EMPTY_MATCHES = True
PROVIDE_SUGGESTIONS = True
# unused GUESS_THRESHOLD = 85.0
# path to the dict
DEFAULT_REGISTER_LOCATION_WORDS = os.path.join(os.path.dirname(ltldoorstep.__file__), '../..', 'tests', 'examples', 'data', 'register-location-words.json')
# creates list of default register locations using the files in the path
# loop is a faster way of creating all the paths
DEFAULT_REGISTER_LOCATIONS = {key: os.path.join(os.path.dirname(ltldoorstep.__file__), '../..', 'tests', 'examples', 'data', f) for key, f in {

    'countries': 'register-countries.json',
    'local-government-district-nir': 'register-geography-local-government-district-nir.json',
    'london-borough-eng': 'register-geography-london-borough-eng.json',
    'metropolitan-district-eng': 'register-geography-metropolitan-district-eng.json',
    'non-metropolitan-district-eng': 'register-geography-non-metropolitan-district-eng.json',
    'registration-district-eng': 'register-geography-registration-district-eng.json',
    'registration-district-wls': 'register-geography-registration-district-wls.json',
    'territory': 'register-geography-territory.json',
    'unitary-authority-wls': 'register-geography-unitary-authority-wls.json',
    'unitary-authority-eng': 'register-geography-unitary-authority-eng.json'
}.items()}

''' Create the matrix of words the processor will look for in the data file
'''
def load_word_matrix(location):
    with open(location, 'r') as word_f:
        word_matrix = json.load(word_f)
    return word_matrix

'''
compares the words in the data file with the words in the matrix and returns the result if there is a match
'''
def find_words(text, word_matrix):
    results = set()
    for word in word_matrix.keys():
        if word in text:
            results.add(word)
    return results

# assesses the metadata?
def assess_metadata(rprt, metadata, word_matrix):
    results = find_words(str(metadata.package), word_matrix)
    return rprt

# creates the table headers (formatting)
def set_properties(df, rprt):
    rprt.set_properties(headers=list(df.columns))
    return rprt

'''
Class to format the data for the countries in the registers 
'''
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
        # Finds matches for the current object & adds the data to the columns
        columns = self.matches & {c.lower() for c in data.columns}
        return columns

    def analyse(self, value):
        # returns the score for the match
        analysis = {'mismatch': value}

        if self.suggest:
            analysis['guess'] = tuple(process.extractOne(value, self.allowed, scorer=fuzz.token_set_ratio))

        return analysis

    def mismatch(self, series):
        return ~series.isin(self.allowed)


"""This is a dict object that contains categories commonly contained within a gov.uk register.
Data passed in will be checked against this in gov_register_checker.
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
        preprocess=lambda result: [entry.replace('citizen', '').strip() for entry in result.split(';')]
    )
]

def find_relevant_columns(data):
    columns = {key: [] for key in data.columns}#list with each key in the columns being searched in

    for col in possible_columns: # loop through each column in the possible columns from the above dict
        for key in col.find_columns(data): # for each key in the data columns, find the data in the columns
            columns[key].append(col)# append the dict column's key to the column

    return {key: entry for key, entry in columns.items() if entry}
# return the data entry for the key, and the entry for the item in the column if it matches the one in the dictionary

def best_matching_series(series, columns):
    series = series.dropna()
    matches = [(col, series[col.mismatch(series)]) for col in columns]

    if EXCLUDE_EMPTY_MATCHES:
        matches = [match for match in matches if len(match[1]) < len(series)]

    if matches:
        col, series = min(matches, key=lambda match_key: len(match_key[1]))
        return col.heading, {key: col.analyse(entry) for key, entry in series.iteritems()}

    return None

# gives report based on json, metadata & registry info. 
def dt_classify_location(data, rprt, metadata, word_matrix):
    overall_results = {} 
    for ix, row in data.iterrows(): # loops through rows in data 
        row_str = ','.join(row.astype(str)).lower() # appends info to row

        # Create a list of codes
        results = {'{}#{}'.format(*entry[0:2]) for rslt in find_words(row_str, word_matrix) for entry in word_matrix[rslt]}

        if results: 
            issue_text = "Possible locations referenced: {}".format(results)
            native_row = json.loads(row.to_json())
            # info to be output for the report
            rprt.add_issue(
                logging.INFO,
                'possible-locations',
                issue_text,
                row_number=ix,
                row=native_row,
                error_data= list(results)
            )

            for result in results:
                if result in overall_results:
                    overall_results[result] += 1

                else:
                    overall_results[result] = 1

    # Use the berlin library to find the LOCODES for the items in overall results
    if 'render-codes' in metadata.configuration and metadata.configuration['render-codes']:
        berlin = load_berlin()
        overall_results = [(berlin.get_code(rslt), n) for rslt, n in overall_results.items()]
        json_results = [(rslt.to_json(), n) for rslt, n in overall_results]
        overall_results = [(rslt.describe(), n) for rslt, n in overall_results] # 
    else:
        overall_results = [(rslt, n) for rslt, n in overall_results.items()]
        json_results = overall_results

    overall_results.sort(key=lambda rslt: rslt[1], reverse=True)
    rows = len(data)
    issue_text = "Overall possible locations referenced: {}".format(', '.join(['{} ({:.2f}%)'.format(rslt, 100 * n / rows) for rslt, n in overall_results]))

    rprt.add_issue(
        logging.INFO,
        'possible-locations-overall',
        issue_text,
        error_data=json_results,
        at_top=True
    )

    # making test_data into data loaded from json
    #register_filename = os.path.join(DEFAULT_REGISTER_LOCATION)

    #with open(register_filename, 'rslt') as data_file:
    #    countries = {row['key']: row['item'][0] for row in json.load(data_file).values()}

    #for col in possible_columns:
    #    col.load_allowed(countries)

    #columns = find_relevant_columns(data)

    #issues = {key: best_matching_series(data[key], entry) for key, entry in columns.items()}

    #matrix = {key: issue for  key, issue in issues.items() if issue}

    #mismatching_columns = {key: issue[1] for key, issue in matrix.items() if issue[1]}
    #for column, issues in mismatching_columns.items():
    #    for row, issue in issues.items():
    #        issue_text = _("Unexpected country-related term '%s'") % issue['mismatch']
    #        if 'guess' in issue and issue['guess'][1] > GUESS_THRESHOLD:
    #            issue_text += _(", perhaps you meant '%s'?") % issue['guess'][0]

    #        column_number = data.columns.get_loc(column)
    #        row_number = data.index.get_loc(row)
    #        native_row = json.loads(data.ix[row].to_json())
    #        rprt.add_issue(
    #            logging.WARNING,
    #            'country-mismatch',
    #            issue_text,
    #            column_number=column_number,
    #            row_number=row_number,
    #            row=native_row,
    #            error_data=issue
    #        )

    #checked_columns = {key: 'country-register-{col}'.format(col=issue[0]) for key, issue in matrix.items()}
    #for column, checker in checked_columns.items():
    #    _("%s using checker %s")
    #    rprt.add_issue(
    #        logging.INFO,
    #        'country-checked',
    #        _("Column %s was checked with state attribute checker %s") % (column, checker),
    #        column_number=data.columns.get_loc(column),
    #        error_data=checked_columns
    #    )

    return rprt # returns the report for each item?

"""
This is the workflow builder. This function will feed the json file into each method and then return the result
"""

class DTClassifyLocationProcessor(DoorstepProcessor):
    @staticmethod
    def make_report():
        return report.TabularReport('datatimes/classify-location:1',("Data Times Location Classification Processor"))

    def get_workflow(self, filename, metadata={}):
        # setting up workflow dict
        workflow = {
            'load_csv': (p.read_csv, filename),
            'load_word_matrix': (load_word_matrix, DEFAULT_REGISTER_LOCATION_WORDS),
            'assess_metadata': (assess_metadata, self._report, self.metadata, 'load_word_matrix'),
            'properties': (set_properties, 'load_csv', 'assess_metadata'),
            'output' : (dt_classify_location, 'load_csv', 'properties', self.metadata, 'load_word_matrix')
        }
        return workflow

processor = DTClassifyLocationProcessor.make

if __name__ == '__main__':
    gettext.install('ltldoorstep')
    proc = processor()

    metadata = {}
    if len(sys.argv) > 2:
        with open(sys.argv[2], 'rslt') as metadata_f:
            metadata = json.load(metadata_f)

    workflow = proc.build_workflow(sys.argv[1], metadata)
    print(get(workflow, 'output'))
