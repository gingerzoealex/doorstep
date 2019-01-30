"""Boundary checking (improved)

This function will (hopefully!) find if data in a csv file is contained within Northern Ireland.
If not so, this will be reported back to the user.
For now, please make sure that the second geojson in the argument is a boundary of Northern Ireland.

"""

import shapely
from geojson_utils import point_in_multipolygon
import logging
import json
from dask.threaded import get
import pandas as p
import geopandas as gp
import csv
import sys
import os
from ltldoorstep.processor import DoorstepProcessor
from ltldoorstep.reports import report

DEFAULT_OUTLINE = 'tests/examples/data/osni-ni-outline-lowres.geojson'


def find_ni_data(first_file, rprt, metadata=None):
    ni_data = DEFAULT_OUTLINE
    if metadata and 'definition' in metadata:
        if metadata and \
                'configuration' in metadata and \
                'boundary' in metadata['configuration'] and \
                metadata['configuration']['boundary'].startswith('$->'):

            boundary_key = metadata['configuration']['boundary'][3:]

            if not 'supplementary' in metadata or boundary_key not in metadata['supplementary']:
                raise RuntimeError("Boundary not found in supplementary data")

            boundary = metadata['supplementary'][boundary_key]
            ni_data = boundary['location']
            rprt.add_supplementary('boundary', boundary['source'], 'Boundary against which points are tested')

            if not os.path.exists(ni_data):
                raise RuntimeError("Boundary not found on filesystem")

    with open(first_file) as data_file:
        # Setting up data that will be compared to the dataset/file being passed in
        data_to_compare = gp.GeoDataFrame.from_features(json.load(data_file)['features'])

    rprt.set_properties(preset='geojson', headers=list(data_to_compare.columns))
    # If csv file has these attributes then...
    if 'geometry' in data_to_compare:

        # This is what we are comparing the first csv/json file to - contains data of NI.
        ni_compare_data = gp.read_file(ni_data)
        # Multipolyon var is set to the first index of ni_compare_data with the key 'geometry'
        multipolygon = ni_compare_data.iloc[0]['geometry']
        # points var is set with data_to_compare with the key 'geometry'
        points = data_to_compare['geometry']
        # outside_points is set to data that is not in multipolygon - this is values outside NI?
        outside_points = data_to_compare[[not multipolygon.contains(p) for p in points]]
        # inside_points_ct is set the sum of the length of points minus outside points
        inside_points_ct = len(points) - len(outside_points)

        # If outside points are not empty then....
        if not outside_points.empty:
            # Iterating through index and points in outside points
            for ix, point in outside_points.iterrows():
                geopoint = shapely.geometry.mapping(point['geometry'])
                # props is set to a dictonary object of point
                props = dict(point)
                # removing key 'geometry'
                del props['geometry']

                # calling Report object method add_issue
                rprt.add_issue(
                    logging.ERROR,
                    'locations-not-found',
                    _("This location is not within the given boundary"),
                    item_index=ix,
                    item=geopoint,
                    item_type=geopoint['type'],
                    item_properties=props
                )
    # If the file does not have any location data....
    else:
        rprt.add_issue(
            'lintol/boundary-checker-improved:1',
            logging.WARNING,
            'no-location-data-found',
            _("No location data found! Please make sure that you have read the right file")
        )

    return rprt

class BoundaryCheckerImprovedProcessor(DoorstepProcessor):
    @staticmethod
    def make_report():
        return report.GeoJSONReport("GeoJSON Boundary Processor", "Info from GeoJSON Processor - example info")

    def get_workflow(self, filename, metadata={}):
        workflow = {
            'output': (find_ni_data, filename, self.make_report(), metadata)
        }
        return workflow

processor = BoundaryCheckerImprovedProcessor

if __name__ == "__main__":
    argv = sys.argv
    processor = BoundaryCheckerImprovedProcessor()
    workflow = processor.build_workflow(argv[1])
    print(compile_report(filename, metadata))
