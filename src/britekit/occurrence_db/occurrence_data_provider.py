#!/usr/bin/env python3

import numpy as np

from britekit.core.exceptions import InputError
from britekit.occurrence_db.occurrence_db import OccurrenceDatabase


class OccurrenceDataProvider:
    """
    Data access layer on top of OccurrenceDatabase.
    If you insert or delete records after creating an instance of this,
    you must call the refresh method.

    Args:
    - db (OccurrenceDatabase): The database object.
    """

    def __init__(self, db: OccurrenceDatabase):
        self.db = db
        self.refresh()

    def refresh(self):
        """Cache database info for quick access"""
        self.counties = self.db.get_all_counties()
        self.classes = self.db.get_all_classes()

        self.county_code_dict: dict[str, object] = {}
        for county in self.counties:
            self.county_code_dict[county.code] = county.id

        self.class_dict: dict[str, object] = {}
        for _class in self.classes:
            self.class_dict[_class.name] = _class

        self.county_dict = {}

    def find_county(self, latitude: float, longitude: float):
        """
        Return county info for a given latitude/longitude, or None if not found.

        Args:
        - latitude (float): Latitude.
        - longitude (float): Longitude.

        Returns:
            County object, or None if not found.
        """

        if (latitude, longitude) in self.county_dict:
            return self.county_dict[(latitude, longitude)]

        for county in self.counties:
            if (
                latitude >= county.min_y
                and latitude <= county.max_y
                and longitude >= county.min_x
                and longitude <= county.max_x
            ):
                # cache for quick access next time
                self.county_dict[(latitude, longitude)] = county
                return county

        return None

    def find_counties(self, region_code: str):
        """
        Return list of counties for a given region code.

        Args:
        - region_code (str): Region code, e.g. "CA", "CA-ON" or "CA-ON-OT".

        Returns:
            List of matching county objects.
        """
        counties = []
        for county in self.counties:
            if county.code.startswith(region_code):
                counties.append(county)

        return counties

    def smoothed_occurrences(self, county_code: str, class_name: str):
        """
        Return list of occurrence values for given county code and class name.
        For each week, return the maximum of it and the adjacent weeks.

        Args:
        - county_code (str): County code
        - class_name (str): Class name

        Returns:
            List of smoothed occurrence values.
        """
        occurrences = self.occurrences(county_code, class_name)
        assert len(occurrences) == 0 or len(occurrences) == 48

        smoothed = np.zeros(len(occurrences))
        for i in range(len(occurrences)):
            smoothed[i] = max(
                max(occurrences[i], occurrences[(i + 1) % 48]),
                occurrences[(i - 1) % 48],
            ).item()

        return smoothed

    def occurrences(self, county_code: str, class_name: str):
        """
        Return list of occurrence values for given county code and class name.

        Args:
        - county_code (str): County code
        - class_name (str): Class name

        Returns:
            List of occurrence values.
        """
        if county_code not in self.county_code_dict:
            raise InputError(f"County {county_code} not found in occurrence database")

        county_id = self.county_code_dict[county_code]
        return self.db.get_occurrences(county_id, class_name)

    def average_occurrences(
        self, county_prefix: str, class_name: str, area_weight: bool = False
    ):
        """
        Given a county code prefix and class name, return the average occurrence values.
        This is used for regional groupings,e.g. county_prefix = "CA" returns the average for Canada
        and "CA-ON" returns the average for Ontario, when eBird county prefixes are used.
        If area_weight = True, weight each county by its area.

        Args:
        - county_prefix (str): County code prefix
        - class_name (str): Class name
        - area_weight (bool, Optional): If true, weight by county area (default = False)

        Returns:
            Numpy array of 48 average occurrence values (one per week, using 4-week months).
        """
        total_values = np.zeros(48)
        zeros = np.zeros(48)
        total_weight = 0
        weight = 1
        for county in self.counties:
            if county.code.startswith(county_prefix):
                curr_values = self.db.get_occurrences(county.id, class_name)
                if area_weight:
                    weight = abs(county.max_x - county.min_x) * abs(
                        county.max_y - county.min_y
                    )

                total_weight += weight
                if len(curr_values) > 0:
                    total_values += np.array(curr_values) * weight
                else:
                    total_values += zeros

        return total_values / total_weight

    def max_occurrences(
        self, county_prefix: str, class_name: str, area_weight: bool = False
    ):
        """
        Given a county code prefix and class name, return the average maximum occurrence value.
        This is used for regional groupings,e.g. county_prefix = "CA" returns the average for Canada
        and "CA-ON" returns the average for Ontario, when eBird county prefixes are used.
        If area_weight = True, weight each county by its area.

        This is not quite the same as average_occurrences.max(), since maximum values in each
        county don't occur in the same week.

        Args:
        - county_prefix (str): County code prefix
        - class_name (str): Class name
        - area_weight (bool, Optional): If true, weight by county area (default = False)

        Returns:
            Numpy average maximum occurrence value.
        """
        import numpy as np

        max_value = 0
        total_weight = 0
        weight = 1
        for county in self.counties:
            if county.code.startswith(county_prefix):
                curr_values = self.db.get_occurrences(county.id, class_name)
                if area_weight:
                    weight = abs(county.max_x - county.min_x) * abs(
                        county.max_y - county.min_y
                    )

                total_weight += weight
                if len(curr_values) > 0:
                    max_value += np.array(curr_values).max() * weight
                else:
                    max_value += 0

        return max_value / total_weight
