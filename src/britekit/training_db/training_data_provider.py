# Defer some imports to improve initialization performance.
from britekit.core import util
from britekit.training_db.training_db import TrainingDatabase


class TrainingDataProvider:
    """
    Data access layer on top of TrainingDatabase.

    Args:
    - db (TrainingDatabase): The database object.
    """

    def __init__(self, db: TrainingDatabase):
        self.db = db

    def category_id(self, name):
        """Return the ID of the specified Category record (insert it if missing)."""
        results = self.db.get_category({"Name": name})
        if not results:
            category_id = self.db.insert_category(name)
        else:
            category_id = results[0].id

        return category_id

    def class_id(self, name, code, category_id):
        """Return the ID of the specified Class record (insert it if missing)."""
        results = self.db.get_class({"Name": name})
        if len(results) == 0:
            class_id = self.db.insert_class(category_id, name, code=code)
        else:
            class_id = results[0].id

        return class_id

    def recording_id(self, class_name, filename, path, source_id, seconds):
        """Return the ID of the specified Recording record (insert it if missing)."""
        results1 = self.db.get_recording_by_class(class_name)
        recording = None
        for r in results1:
            if r.source_id == source_id and r.filename == filename:
                recording = r
                break

        if recording is None:
            recording_id = self.db.insert_recording(source_id, filename, path, seconds)
        else:
            recording_id = recording.id

        return recording_id

    def specgroup_id(self, name):
        """Return the ID of the specified SpecGroup record (insert it if missing)."""
        results = self.db.get_specgroup({"Name": name})
        if len(results) == 0:
            specgroup_id = self.db.insert_specgroup(name)
        else:
            specgroup_id = results[0].id

        return specgroup_id

    def source_id(self, filename, source_name=None):
        """Return the ID of the specified Source record (insert it if missing)."""
        if source_name is None:
            source_name = util.get_source_name(filename)

        results = self.db.get_source({"Name": source_name})
        if len(results) == 0:
            source_id = self.db.insert_source(source_name)
        else:
            source_id = results[0].id

        return source_id

    def class_info(self):
        """
        Get a summary and details dataframe with class, recording and segment counts.

        Returns:
            summary_df: A pandas dataframe with recording and segment counts per class
            details_df: A pandas dataframe with segment counts per recording per class
        """
        import pandas as pd

        classes = self.db.get_class()
        recordings_per_class = {}
        names, codes, recording_count, segment_count = [], [], [], []
        for c in sorted(classes, key=lambda obj: obj.name):
            names.append(c.name)
            codes.append(c.code)
            results = self.db.get_recording_by_class(c.name)
            recordings_per_class[c.name] = results
            recording_count.append(len(results))

        # this is much faster than getting the count once per class
        results = self.db.get_all_segment_counts()
        for r in results:
            segment_count.append(r.count)

        summary_df = pd.DataFrame()
        summary_df["name"] = names
        summary_df["code"] = codes
        summary_df["recordings"] = recording_count
        summary_df["segments"] = segment_count

        names, codes, recordings, segment_count = [], [], [], []
        for c in sorted(classes, key=lambda obj: obj.name):
            for r in sorted(recordings_per_class[c.name], key=lambda obj: obj.filename):
                names.append(c.name)
                codes.append(c.code)
                recordings.append(r.filename)
                segment_count.append(self.db.get_segment_count({"RecordingID": r.id}))

        details_df = pd.DataFrame()
        details_df["name"] = names
        details_df["code"] = codes
        details_df["recording"] = recordings
        details_df["segments"] = segment_count

        return summary_df, details_df

    def spec_group_info(self):
        """
        Get a dataframe with number of spectrograms per spec group.

        Returns:
            A pandas dataframe with number of spectrograms per spec group.
        """
        import pandas as pd

        names = []
        counts = []
        spec_groups = self.db.get_specgroup()
        for s in spec_groups:
            names.append(s.name)
            counts.append(self.db.get_specvalue_count({"SpecGroupID": s.id}))

        df = pd.DataFrame()
        df["name"] = names
        df["spectrograms"] = counts
        return df

    def segment_class_dict(self):
        """
        Get a dict from segment ID to a set of class names.

        Returns:
            dict from segment ID to a set of class names
        """

        # map class ID to name
        class_id_dict = {}
        results = self.db.get_class()
        for r in results:
            class_id_dict[r.id] = r.name

        # map spec ID to set of class names
        segment_class_dict: dict[int, set] = {}
        results = self.db.get_segment_class()
        for r in results:
            if r.segment_id not in segment_class_dict:
                segment_class_dict[r.segment_id] = set()

            segment_class_dict[r.segment_id].add(class_id_dict[r.class_id])

        return segment_class_dict
