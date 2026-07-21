#!/usr/bin/env python3

import os
import pickle
from types import SimpleNamespace
from typing import Optional

from britekit.occurrence_db.occurrence_pickle_v2 import FORMAT_NAME, FORMAT_VERSION


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

        if isinstance(self.data, dict) and self.data.get("format") == FORMAT_NAME:
            version = self.data.get("version")
            if version != FORMAT_VERSION:
                raise ValueError(
                    f"Unsupported occurrence pickle version {version}; "
                    f"expected {FORMAT_VERSION}"
                )
            self._initialize_v2()
        else:
            self.format_version = 1
            self.counties = list(self.data["counties"].values())
            self.class_names = set(self.data["classes"])

        self.county_cache = {}  # cache (latitude/longitude) -> county
        self.occurrences = {}  # cache occurrence values in a region
        self.smoothed = {}  # cache smoothed values in a region
        self.max = {}  # cache max values in a region

    def _initialize_v2(self) -> None:
        import numpy as np

        try:
            self.format_version = FORMAT_VERSION
            self.counties = [
                SimpleNamespace(
                    index=index,
                    code=area[0],
                    name=area[1],
                    min_x=area[2],
                    max_x=area[3],
                    min_y=area[4],
                    max_y=area[5],
                )
                for index, area in enumerate(self.data["areas"])
            ]
            self.class_names = set(self.data["class_names"])
            self.class_name_to_index = {
                name: index for index, name in enumerate(self.data["class_names"])
            }
            self.area_offsets = np.asarray(self.data["area_offsets"], dtype=np.int64)
            self.class_indices = np.asarray(self.data["class_indices"], dtype=np.int32)
            self.values = np.asarray(self.data["values"], dtype=np.float16)
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise ValueError(
                f"Invalid occurrence pickle v{FORMAT_VERSION}: {error}"
            ) from error

        if self.values.ndim != 2 or self.values.shape[1] != 48:
            raise ValueError("Invalid occurrence pickle v2 weekly-value matrix")
        if len(self.area_offsets) != len(self.counties) + 1:
            raise ValueError("Invalid occurrence pickle v2 area offsets")
        if len(self.class_indices) != len(self.values):
            raise ValueError("Invalid occurrence pickle v2 occurrence indexes")

    def _occurrence_values(self, county, class_name):  # type: ignore[no-untyped-def]
        """Return raw values for one area/class, or None when absent."""
        if self.format_version == 1:
            return self.data["occurrences"][county.code].get(class_name)

        import numpy as np

        class_index = self.class_name_to_index.get(class_name)
        if class_index is None:
            return None
        start = self.area_offsets[county.index]
        end = self.area_offsets[county.index + 1]
        relative = np.searchsorted(self.class_indices[start:end], class_index)
        position = start + relative
        if position >= end or self.class_indices[position] != class_index:
            return None
        return self.values[position]

    def _maximum_value(self, county, class_name, occurrence_values):  # type: ignore[no-untyped-def]
        if self.format_version == 1:
            return self.data["max"][county.code][class_name]

        return occurrence_values.max().item()

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

        for county in self.counties:
            if (
                county.min_y is not None
                and county.max_y is not None
                and county.min_x is not None
                and county.max_x is not None
                and latitude >= county.min_y
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
        for county in self.counties:
            if county.code.startswith(region_code):
                counties.append(county)

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

        class_found = class_name in self.class_names
        if not location_found or not class_found:
            return location_found, class_found, None

        if len(counties) == 1:
            occurrence_values = self._occurrence_values(counties[0], class_name)
            if occurrence_values is not None:
                if week_num is None:
                    if counties[0].code not in self.max:
                        self.max[counties[0].code] = {}
                    value = self._maximum_value(
                        counties[0], class_name, occurrence_values
                    )
                    self.max[counties[0].code][class_name] = value
                    return True, True, value
                elif smoothed:
                    if counties[0].code not in self.smoothed:
                        self.smoothed[counties[0].code] = {}
                    values = (
                        self.data["smoothed"][counties[0].code][class_name]
                        if self.format_version == 1
                        else np.maximum(
                            occurrence_values,
                            np.maximum(
                                np.roll(occurrence_values, 1),
                                np.roll(occurrence_values, -1),
                            ),
                        )
                    )
                    self.smoothed[counties[0].code][class_name] = values
                    return (
                        True,
                        True,
                        values[week_num],
                    )
                else:
                    if counties[0].code not in self.occurrences:
                        self.occurrences[counties[0].code] = {}
                    self.occurrences[counties[0].code][class_name] = occurrence_values
                    return (
                        True,
                        True,
                        occurrence_values[week_num],
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
                occurrence_values = self._occurrence_values(county, class_name)
                if occurrence_values is not None:
                    matches += 1
                    if week_num is None:
                        max_val += self._maximum_value(
                            county, class_name, occurrence_values
                        )
                    elif smoothed:
                        if self.format_version == 1:
                            smoothed_vals += self.data["smoothed"][county.code][
                                class_name
                            ]
                        else:
                            smoothed_vals += np.maximum(
                                occurrence_values,
                                np.maximum(
                                    np.roll(occurrence_values, 1),
                                    np.roll(occurrence_values, -1),
                                ),
                            )
                    else:
                        occurrences += occurrence_values

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
