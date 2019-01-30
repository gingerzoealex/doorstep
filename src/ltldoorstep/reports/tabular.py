from .report import Report, ReportItem
import itertools

class TabularReport(Report):

    preset = 'tabular'

    def add_issue(self, log_level, code, message, row_number=None, column_number=None, row=None, error_data=None, at_top=False):
        """This function will add an issue to the report and takes as parameters the processor, the log level, code, message"""

        if row and self.properties['headers']:
            if isinstance(row, dict):
                base = {k: None for k in self.properties['headers']}
                base.update(row)
                row = base
            else:
                row_pairs = itertools.zip_longest(self.properties['headers'], row)
                row = {k: v for k, v in row_pairs if k is not None}

        context = None
        properties = None
        if row_number:
            if column_number:
                typ = 'Cell'
                if row:
                    context = [ReportItem('Row', {'row': row_number, 'column': None}, None, row)]
            else:
                typ = 'Row'
                properties = row
        else:
            if column_number:
                typ = 'Column'
            else:
                typ = 'Global'

        item = ReportItem(typ, {'row': row_number, 'column': column_number}, None, properties)

        super(TabularReport, self).add_issue(log_level, code, message, item, error_data=error_data, context=context, at_top=at_top)
