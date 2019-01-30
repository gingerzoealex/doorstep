import os
import logging

from .reports.report import Report, get_report_class_from_preset, combine_reports
from .metadata import DoorstepMetadata

class DoorstepProcessor:
    preset = None
    code = None
    description = None
    _metadata = None

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        if type(metadata) is dict:
            metadata = DoorstepMetadata.from_dict(metadata)
        self._metadata = metadata

    @classmethod
    def make_report(cls):
        report = get_report_class_from_preset(cls.preset)

        if cls.code:
            code = cls.code
        else:
            code = _("(unknown processor)")

        if cls.description:
            description = cls.description
        else:
            description = _("(no processor description provided)")

        return report(code, description)

    def initialize(self, report=None, metadata=None):
        if report is None:
            report = self.make_report()
        self._report = report
        self.metadata = metadata

    @classmethod
    def make(cls):
        new = cls()
        new.initialize()
        return new

    def compile_report(self, filename=None, metadata=None):
        return self._report.compile(filename, metadata)

    def get_report(self):
        return self._report

    def set_report(self, report):
        self._report = report

    def build_workflow(self, filename, metadata={}):
        self.metadata = metadata
        return self.get_workflow(filename, metadata)

    def get_workflow(self, filename, metadata={}):
        return {}
