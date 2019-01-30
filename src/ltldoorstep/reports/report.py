"""This is the report object that will be used for the standardisation of processors reporting
The Report superclass can be inherited for different forms of reporting i.e. tabular, GeoJSON etc."""


import logging
import json
import os

def get_report_class_from_preset(preset):
    if preset not in _report_class_from_preset:
        raise NotImplementedError(_(
            "This version of doorstep does not have the requested Report preset type"
        ))
    return _report_class_from_preset[preset]

class ReportItem:
    def __init__(self, typ, location, definition, properties):
        self.type = typ
        self.location = location
        self.definition = definition
        self.properties = dict(properties) if properties else None

    def __str__(self):
        return json.dumps(self.render())

    def __repr__(self):
        return '(|Item: %s|)' % str(self)

    @classmethod
    def parse(cls, data):
        if data['entity']:
            return cls(
                typ=data['entity']['type'] if data['entity'] else None,
                location=data['entity']['location'] if data['entity'] else None,
                definition=data['entity']['definition'] if data['entity'] else None,
                properties=data['properties']
            )
        else:
            return cls(properties=data['properties'])

    def render(self):
        return {
            'entity': {
                'type': self.type,
                'location': self.location,
                'definition': self.definition
            },
            'properties': self.properties
        }

class ReportItemLiteral(ReportItem):
    def __init__(self, literal):
        self.literal = literal

    def render(self):
        return self.literal

class ReportIssue:
    def __init__(self, level, item, context, processor, code, message, error_data=None):
        self.level = level
        self.processor = processor
        self.code = code
        self.message = message
        self.item = item
        self.context = context

        if error_data is None:
            error_data = {}
        self.error_data = error_data

    def __str__(self):
        return json.dumps(self.render())

    def __repr__(self):
        return '(|Issue: %s|)' % str(self)

    def get_item(self):
        return self.item

    @classmethod
    def parse(cls, level, data):
        return cls(
            level=level,
            processor=data['processor'],
            code=data['code'],
            message=data['message'],
            item=ReportItem.parse(data['item']),
            context=[ReportItem.parse(c) for c in data['context']] if data['context'] else None,
            error_data=(data['error-data'] if 'error-data' in data else {})
        )

    def render(self):
        return {
            'processor': self.processor,
            'code' : self.code,
            'message' : self.message,
            'item': self.item.render(),
            'context': [c.render() for c in self.context] if self.context else None,
            'error-data': self.error_data
        }

class ReportIssueLiteral(ReportIssue):
    def __init__(self, level, literal, literal_item):
        self.level = level
        self.literal = literal
        self.item = ReportItemLiteral(literal_item)

    def render(self):
        return self.literal

class Report:
    preset = None

    @classmethod
    def get_preset(cls):
        if not cls.preset:
            raise NotImplementedError(_("This report class must have a preset type"))

        return cls.preset

    def __str__(self):
        return json.dumps(self.compile())

    def __repr__(self):
        return '(|Report: %s|)' % str(self)

    def __init__(self, processor, info, filename='', metadata=None, headers=None, encoding='utf-8', time=0., row_count=None, supplementary=None, issues=None):
        if issues is None:
            self.issues = {
                logging.ERROR: [],
                logging.WARNING: [],
                logging.INFO: []
            }
        else:
            self.issues = issues

        if metadata is None:
            metadata = {}
        if supplementary is None:
            supplementary = []

        self.processor = processor
        self.info = info
        self.supplementary = supplementary
        self.filename = filename
        self.metadata = metadata

        self.properties = {
            'row-count': row_count,
            'time': time,
            'encoding': encoding,
            'preset': self.get_preset(),
            'headers': headers
        }

    def get_issues(self, level=None):
        if level:
            return self.issues[level]
        return sum(list(self.issues.values()))

    def append_issue(self, issue, prepend=False):
        if prepend:
            self.issues[issue.level].insert(0, issue)
        else:
            self.issues[issue.level].append(issue)

    @classmethod
    def load(cls, file_obj):
        return cls.parse(json.load(file_obj))

    @classmethod
    def parse(cls, dictionary):
        issues = {}

        table = dictionary['tables'][0]

        issues[logging.ERROR] = [
            ReportIssue.parse(logging.ERROR, issue)
            for issue in
            table['errors']
        ]

        issues[logging.WARNING] = [
            ReportIssue.parse(logging.WARNING, issue)
            for issue in
            table['warnings']
        ]

        issues[logging.INFO] = [
            ReportIssue.parse(logging.INFO, issue)
            for issue in
            table['informations']
        ]

        filename = dictionary['filename']
        metadata = {
            'fileType': table['format']
        }
        supplementary = dictionary['supplementary']
        logging.warning(supplementary)
        row_count = table['row-count'] if 'time' in table else None
        time = table['time'] if 'time' in table else None
        encoding = table['encoding'] if 'encoding' in table else None
        headers = table['headers'] if 'headers' in table else None

        cls = get_report_class_from_preset(dictionary['preset'])

        return cls(
            '(unknown)',
            '(parsed by ltldoorstep)',
            filename=filename,
            supplementary=supplementary,
            row_count=row_count,
            time=time,
            encoding=encoding,
            headers=headers,
            metadata=metadata,
            issues=issues
        )

    def update(self, additional):
        for level, issues in additional.issues.items():
            self.issues[level] += issues

        self.supplementary += additional.supplementary

        for prop in ('row-count', 'encoding', 'headers', 'preset'):
            if self.properties[prop] is None:
                self.properties[prop] = additional.properties[prop]

    def compile(self, filename=None, metadata=None):
        report = {k: [item.render() for item in v] for k, v in self.issues.items()}
        supplementary = self.supplementary

        if filename is None:
            filename = self.filename

        if not filename:
            filename = 'unknown.csv'

        if metadata is None:
            metadata = self.metadata

        if metadata and 'fileType' in metadata:
            frmt = metadata['fileType']
        else:
            root, frmt = os.path.splitext(filename)
            if frmt and frmt[0] == '.':
                frmt = frmt[1:]

        valid = not bool(report[logging.ERROR])

        return {
            'supplementary': supplementary,
            'error-count': sum([len(r) for r in report.values()]),
            'valid': valid,
            'filename': filename,
            'tables': [
                {
                    'format': frmt,
                    'errors': report[logging.ERROR],
                    'warnings': report[logging.WARNING],
                    'informations': report[logging.INFO],
                    'row-count': self.properties['row-count'],
                    'headers': self.properties['headers'],
                    'source': filename,
                    'time': self.properties['time'],
                    'valid': valid,
                    'scheme': 'file',
                    'encoding': self.properties['encoding'],
                    'schema': None,
                    'error-count': sum([len(r) for r in report.values()])
                }
            ],
            'preset': self.get_preset(),
            'warnings': [],
            'table-count': 1,
            'time': self.properties['time']
        }

    def add_supplementary(self, typ, source, name):
        logging.warning('Adding supplementary')
        logging.warning((typ, source, name))
        self.supplementary.append({
            'type': typ,
            'source': source,
            'name': name
        })

    def set_properties(self, **properties):
        for arg in self.properties:
            if arg in properties:
                self.properties[arg] = properties[arg]


    def add_issue(self, log_level, code, message, item=None, error_data=None, context=None, at_top=False):
        """This function will add an issue to the report and takes as parameters the processor, the log level, code, message"""


        if log_level not in self.issues:
            raise RuntimeError(_('Log-level must be one of logging.INFO, logging.WARNING or logging.ERROR'))

        if not isinstance(item, ReportItem):
            item = ReportItemLiteral(item)

        issue = ReportIssue(log_level, item=item, context=context, processor=self.processor, code=code, message=message, error_data=error_data)

        self.append_issue(issue, prepend=at_top)

def properties_from_report(report):
    table = report['tables'][0]
    return {
        'row-count': table['row-count'],
        'time': report['time'],
        'encoding': table['encoding'],
        'preset': report['preset'],
        'headers': table['headers']
    }

def combine_reports(*reports, base=None):
    if base is None:
        presets = {report.preset for report in reports if report.preset}

        if len(presets) != 1:
            raise RuntimeError(
                _("Report combining can only be performed on reports with the same 'preset' property")
            )
        preset = list(presets)[0]
        rcls = get_report_class_from_preset(preset)
        base = rcls(None, None)

    for report in reports:
        base.update(report)

    return base


from .geojson import GeoJSONReport
from .tabular import TabularReport

_report_class_from_preset = {
    cls.get_preset(): cls for cls in
    (GeoJSONReport, TabularReport)
}
