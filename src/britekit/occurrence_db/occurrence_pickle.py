import os
import pickle
from typing import Optional


class OccurrencePickleProvider:
    def __init__(self, pickle_path=None):
        if pickle_path is None:
            pickle_path = os.path.join("data", "occurrence.pkl")

        try:
            with open(pickle_path, "rb") as f:
                self.data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
        except (pickle.UnpicklingError, EOFError) as e:
            raise ValueError(f"Failed to load pickle file {pickle_path}: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error loading pickle file {pickle_path}: {e}")

        self.county_cache = {}  # cache (latitude/longitude) -> county
        self.occurrences = {}  # cache occurrence values in a region
        self.smoothed = {}  # cache smoothed values in a region
        self.max = {}  # cache max values in a region

    def find_county(self, latitude: float, longitude: float):
        """
        Return county info for a given latitude/longitude, or None if not found.

        Args:
        - latitude (float): Latitude.
        - longitude (float): Longitude.

        Returns:
            County object, or None if not found.
        """

        if (latitude, longitude) in self.county_cache:
            return self.county_cache[(latitude, longitude)]

        for county_code in self.data["counties"]:
            county = self.data["counties"][county_code]
            if (
                latitude >= county.min_y
                and latitude <= county.max_y
                and longitude >= county.min_x
                and longitude <= county.max_x
            ):
                # cache for quick access next time
                self.county_cache[(latitude, longitude)] = county
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
        for county_code in self.data["counties"]:
            if county_code.startswith(region_code):
                counties.append(self.data["counties"][county_code])

        return counties

    def occurrence_value(
        self,
        class_name: str,
        smoothed: bool = True,
        region_code: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        week_num: Optional[int] = None,
    ):
        """
        Given a class name, region code or latitude/longitude, and optional week number,
        return the occurrence value for the given class/location/week.
        Given a week and multiple counties, return the average value across counties.
        If no week is given, return the max value.

        Args:
        - class_name (str): Class name
        - smoothed (bool): If true, use the max of adjacent week's values for each week.
        - region_code (str, optional): Region code. If omitted, latitude/longitude must be intovided.
        - latitude (float, optional): Latitude
        - longitude (float, optional): Longitude
        - week_num (int, optional):

        Returns:
        - location_found (bool): True iff region/lat/lon map to a known county or counties
        - class_found (bool): True iff class_name is in occurrence database
        - occurrence (float): If location_found and class_found, occurrence value for given class/location/week, else None
        """
        import numpy as np

        assert region_code is not None or (
            latitude is not None and longitude is not None
        )
        if week_num is not None:
            assert week_num >= 0 and week_num <= 47

        location_found = True
        if region_code is None:
            assert latitude is not None and longitude is not None
            county = self.find_county(latitude, longitude)
            if county is None:
                location_found = False
            else:
                counties = [county]
        else:
            # use cached value if possible
            if week_num is None:
                if region_code in self.max and class_name in self.max[region_code]:
                    return True, True, self.max[region_code][class_name]
            elif smoothed:
                if (
                    region_code in self.smoothed
                    and class_name in self.smoothed[region_code]
                ):
                    return True, True, self.smoothed[region_code][class_name][week_num]
            elif (
                region_code in self.occurrences
                and class_name in self.occurrences[region_code]
            ):
                return True, True, self.occurrences[region_code][class_name][week_num]

            # not found in cache
            counties = self.find_counties(region_code)
            if len(counties) == 0:
                location_found = False

        class_found = class_name in self.data["classes"]
        if not location_found or not class_found:
            return location_found, class_found, None

        if len(counties) == 1:
            if class_name in self.data["occurrences"][counties[0].code]:
                if week_num is None:
                    if counties[0].code not in self.max:
                        self.max[counties[0].code] = {}
                    self.max[counties[0].code][class_name] = self.data["max"][
                        counties[0].code
                    ][class_name]
                    return True, True, self.data["max"][counties[0].code][class_name]
                elif smoothed:
                    if counties[0].code not in self.smoothed:
                        self.smoothed[counties[0].code] = {}
                    self.smoothed[counties[0].code][class_name] = self.data["smoothed"][
                        counties[0].code
                    ][class_name]
                    return (
                        True,
                        True,
                        self.data["smoothed"][counties[0].code][class_name][week_num],
                    )
                else:
                    if counties[0].code not in self.occurrences:
                        self.occurrences[counties[0].code] = {}
                    self.occurrences[counties[0].code][class_name] = self.data[
                        "occurrences"
                    ][counties[0].code][class_name]
                    return (
                        True,
                        True,
                        self.data["occurrences"][counties[0].code][class_name][
                            week_num
                        ],
                    )
            else:
                return True, False, None
        else:
            # multi-county region
            occurrences = np.zeros(48)
            smoothed_vals = np.zeros(48)
            max_val = 0.0
            matches = 0
            for county in counties:
                if class_name in self.data["occurrences"][county.code]:
                    matches += 1
                    if week_num is None:
                        max_val += self.data["max"][county.code][class_name]
                    elif smoothed:
                        smoothed_vals += self.data["smoothed"][county.code][class_name]
                    else:
                        occurrences += self.data["occurrences"][county.code][class_name]

            if matches > 0:
                if week_num is None:
                    max_val /= matches
                    if region_code not in self.max:
                        self.max[region_code] = {}
                    self.max[region_code][class_name] = max_val
                    return True, True, max_val
                elif smoothed:
                    smoothed_vals /= matches
                    if region_code not in self.smoothed:
                        self.smoothed[region_code] = {}
                    self.smoothed[region_code][class_name] = smoothed_vals
                    return True, True, smoothed_vals[week_num]
                else:
                    occurrences /= matches
                    if region_code not in self.occurrences:
                        self.occurrences[region_code] = {}
                    self.occurrences[region_code][class_name] = occurrences
                    return True, True, occurrences[week_num]
            else:
                # class exists but not in any of the counties
                return True, False, None
