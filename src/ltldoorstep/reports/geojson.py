from .report import Report

class GeoJSONReport(Report):

    preset = 'geojson'

    def add_issue(self, log_level, code, message, item_index=None, item=None, item_type=None,
                  item_properties=None, error_data=None, at_top=False):
        """This function will add an issue to the report and takes as parameters the processor, the log level, code, message"""

        if item:
            item = {
                'entity': {
                    'type': item_type,
                    'location': {
                        'index': item_index
                    },
                    'definition': item
                },
                'properties': item_properties
            }

        super(GeoJSONReport, self).add_issue(log_level, code, message, item, error_data, at_top=at_top)
