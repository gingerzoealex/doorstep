import sys
import os
import json
import pandas
import ltldoorstep
import berlin
import berlin.backend_dict
import berlin.multicode
import logging
import re
from itertools import chain

DIR = os.path.dirname(__file__)
CODE_ISO3166_1_ALPHA2 = 0
CODE_ISO3166_2 = 1
CODE_UN_LOCODE = 2

re_parens = re.compile(r'\((.*)\)')
re_brackets = re.compile(r'\[(.*)\]')
re_extra_codes = re.compile(r'[A-Z]{2}[:-][A-Z]{3}')
re_dividers = re.compile(r'(?: and |,)', re.I)
re_obstacles = re.compile(r'(?: the )', re.I)

def extract_names(names):
    new_names = set()
    for name in names:
        matches = chain(re_parens.finditer(name), re_brackets.finditer(name))
        name = re_parens.sub('', name)
        name = re_brackets.sub('', name)
        name = re_extra_codes.sub('', name)
        name = re_obstacles.sub('', name)

        split = re_dividers.split(name)
        if len(split) > 1:
            print(split, '<<<')
            new_names = new_names.union(extract_names(split))

        new_names.add(name.strip())
        new_names = new_names.union(extract_names({match.group(1) for match in matches}))

    new_names = new_names.union({n.encode('ascii', 'ignore').decode() for n in new_names})
    if '' in new_names:
        new_names.remove('')

    return new_names

class RegisterExtractor:
    data_sources = {
        'iata_file': 'locode/airport-codes/data/airport-codes.csv',
        'state_file': 'locode/country-codes/data/country-codes.csv',
        'subdiv_file': [
            'locode/un-locode/data/subdivision-codes.csv',
            'locode/iso-3166-2/iso_3166_2.js',
        ],
        'locode_file': 'locode/un-locode/data/code-list.csv'
    }
    parser_states = {}
    missing_codes = {}
    missing_names = {}

    @classmethod
    def get_parser(cls):
        backend = berlin.backend_dict.BackendDict()

        cls.code_dict = backend.retrieve(cls.data_sources, progress_bar=True)
        cls.parser = [berlin.multicode.RegionParser(d, distances=False) for d in cls.code_dict[0:2]]

    @classmethod
    def load_patch(cls):
        patch = pandas.read_csv(os.path.join(DIR, 'register-patch.csv'))
        for ix, row in patch.iterrows():
            if not pandas.isna(row['code']) and not pandas.isna(row['code_type']):
                replacement = None
                if not pandas.isna(row['replacement_code_type']) and not pandas.isna(row['replacement_code']):
                    replacement = (row['replacement_code_type'], row['replacement_code'])
                cls.missing_codes[(row['code_type'], row['code'])] = replacement

            if not pandas.isna(row['state']) and not pandas.isna(row['name']):
                replacement = None
                if not pandas.isna(row['replacement_code_type']) and not pandas.isna(row['replacement_code']):
                    replacement = (row['replacement_code_type'], row['replacement_code'])
                cls.missing_names[(row['state'], row['name'])] = replacement

    @classmethod
    def check_patch_by_code(cls, code_type, code):
        if (code_type, code) in cls.missing_codes:
            replacement = cls.missing_codes[(code_type, code)]
            if replacement:
                replacement_code_type, replacement_code = replacement
                return cls.code_dict[replacement_code_type][replacement_code]
            else:
                return True
        return None

    @classmethod
    def check_patch_by_name(cls, state, name):
        if (state, name) in cls.missing_names:
            replacement = cls.missing_names[(state, name)]
            if replacement:
                replacement_code_type, replacement_code = replacement
                return cls.code_dict[replacement_code_type][replacement_code]
            else:
                return True
        return None

    @classmethod
    def get_parser_state(cls, state):
        if state not in cls.parser_states:
            parser = berlin.multicode.RegionParser(cls.code_dict[3][state], distances=False)
            cls.parser_states[state] = parser
        return cls.parser_states[state]

    def extract(self):
        if self.filename.endswith('.json'):
            with open(self.filename, 'r') as register_f:
                rdata = json.load(register_f)

            iterator = rdata.items()
        else:
            rdata = pandas.read_csv(self.filename, na_filter=False)
            iterator = rdata.iterrows()

        for key, row in iterator:
            try:
                code = self.key_extractor(key, row)
                code_type = code[0]
                expected_locode = code[1]

                locode = None
                if expected_locode:
                    replacement = self.check_patch_by_code(code_type, expected_locode)
                    if replacement:
                        if replacement is True:
                            continue
                        else:
                            locode = replacement
                    elif expected_locode in self.code_dict[code_type]:
                        locode = self.code_dict[code_type][expected_locode]

                if not locode and len(code) > 2:
                    state, phrase = code[2]

                    replacement = self.check_patch_by_name(state, phrase)
                    if replacement:
                        if replacement is True:
                            continue
                        else:
                            locode = replacement
                    # Catch any non-subdivisions that are found in locode list
                    elif not locode and expected_locode in self.code_dict[3][state]:
                        locode = self.code_dict[3][state][expected_locode]
                    else:
                        parsers = (self.parser[code_type], self.get_parser_state(state))

                        for parser in parsers:
                            locode_tuple = parser.analyse(matches=1, name=code[2][1], supercode=code[2][0])
                            score = locode_tuple[2]

                            if score > 1.5:
                                locode = locode_tuple[1]
                                break

                #assert locode, (code_type, code[1], row)
                if not locode:
                    print('***', code_type, expected_locode, row)
                    continue

                yield (
                    key,
                    locode,
                    {e.lower() for e in self.value_extractor(locode, row)}
                )
            except AssertionError as e:
                logging.error("Error for key {} in {}: {}".format(key, self.name, str(e)))
                raise e

    def __init__(self, name, filename, key_extractor, value_extractor):
        self.name = name
        self.filename = os.path.join(DIR, filename)
        self.key_extractor = key_extractor
        self.value_extractor = value_extractor

def get_register_locations():
    return {r.name: r for r in [
        RegisterExtractor(
            'countries',
            'register-countries.json',
            lambda k, r: (CODE_ISO3166_1_ALPHA2, k),
            lambda l, r: extract_names([r['item'][0]['name']] + r['item'][0]['citizen-names'].split(';'))
        ),
        RegisterExtractor(
            'local-government-district-nir',
            'register-geography-local-government-district-nir.csv',
            lambda k, r: (CODE_ISO3166_2, 'GB:' + r['local-authority-nir']),
            lambda l, r: extract_names([l.name])
        ),
        RegisterExtractor(
            'london-borough-eng',
            'register-geography-london-borough-eng.csv',
            lambda k, r: (CODE_ISO3166_2, 'GB:' + r['local-authority-eng'], ('GB', r['local-authority-eng'])),
            lambda l, r: extract_names([l.name])
        ),
        RegisterExtractor(
            'metropolitan-district-eng',
            'register-geography-metropolitan-district-eng.csv',
            lambda k, r: (CODE_ISO3166_2, 'GB:' + r['local-authority-eng'], ('GB', r['local-authority-eng'])),
            lambda l, r: extract_names([l.name])
        ),
        RegisterExtractor(
            'non-metropolitan-district-eng',
            'register-geography-non-metropolitan-district-eng.csv',
            lambda k, r: (CODE_ISO3166_2, 'GB:' + r['local-authority-eng'], ('GB', r['name'])),
            lambda l, r: extract_names([l.name, r['name']])
        ),
        RegisterExtractor(
            'territory',
            'register-geography-territory.csv',
            lambda k, r: (CODE_ISO3166_2 if '-' in r['key'] else CODE_ISO3166_1_ALPHA2, r['key'].replace('-', ':')),
            lambda l, r: extract_names([l.name, r['name']])
        ),
        RegisterExtractor(
            'unitary-authority-eng',
            'register-geography-unitary-authority-eng.csv',
            lambda k, r: (CODE_ISO3166_2, 'GB:' + r['local-authority-eng']),
            lambda l, r: extract_names([l.name])
        ),
        RegisterExtractor(
            'unitary-authority-wls',
            'register-geography-unitary-authority-wls.csv',
            lambda k, r: (CODE_ISO3166_2, 'GB:' + r['principal-local-authority']),
            lambda l, r: extract_names([l.name])
        )
    ]}


def run():
    RegisterExtractor.get_parser()
    RegisterExtractor.load_patch()

    logging.info("Parsed Berlin")

    words_matrix = {}

    register_locations = get_register_locations()
    for register in register_locations.values():
        print()
        print(register.name)
        print('=' * len(register.name))

        entry_count = 0
        for key, locode, entries in register.extract():
            for entry in entries:
                if entry not in words_matrix:
                    words_matrix[entry] = []
                words_matrix[entry].append((locode.code_type, locode.identifier, register.name, key))
            entry_count += len(entries)

        print('Found {:d} entries'.format(entry_count))

    print('Total entries is {:d}'.format(len(words_matrix)))

    with open('register-location-words.json', 'w') as words_f:
        json.dump(words_matrix, words_f)

if __name__ == "__main__":
    run()
