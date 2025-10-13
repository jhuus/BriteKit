## API Reference
### Classes
- [Analyzer](#analyzer)
- [Audio](#audio)
- [Extractor](#extractor)
- [OccurrenceDataProvider](#occurrencedataprovider)
- [OccurrenceDatabase](#occurrencedatabase)
- [PerMinuteTester](#perminutetester)
- [PerRecordingTester](#perrecordingtester)
- [PerSegmentTester](#persegmenttester)
- [Pickler](#pickler)
- [Predictor](#predictor)
- [Trainer](#trainer)
- [TrainingDataProvider](#trainingdataprovider)
- [TrainingDatabase](#trainingdatabase)
- [Tuner](#tuner)

### Analyzer
**Class**  
```python
Analyzer()
```
Basic inference logic using Predictor class, with multi-threading and multi-recording support.

**Public methods & properties**

**run**  
```python
Analyzer.run(self, input_path: str, output_path: str, rtype: str = 'audacity')
```
Run inference.

Args:

- `input_path` *(str)* — Recording or directory containing recordings.
- `output_path` *(str)* — Output directory.
- `rtype` *(str)* — Output format: "audacity", "csv" or "both".



### Audio
**Class**  
```python
Audio(device: Optional[str] = None, cfg: Optional[britekit.core.base_config.BaseConfig] = None)
```
Provide methods for reading audio recordings and creating spectrograms.

Attributes:
    device (str, optional): Device supported by pytorch ('cuda', 'cpu' or 'mps')
    cfg (optional): BaseConfig to use.

**Public methods & properties**

**get_spectrograms**  
```python
Audio.get_spectrograms(self, start_times: list[float], spec_duration: Optional[float] = None, freq_scale: Optional[str] = None, decibels: Optional[float] = None, top_db: Optional[int] = None, db_power: Optional[int] = None)
```
Generate normalized and unnormalized spectrograms for specified time offsets.

Creates spectrograms from the loaded audio signal at the specified start times.
Supports different frequency scales (linear, log, mel) and optional decibel conversion.
Returns both normalized (0-1 range) and unnormalized versions of the spectrograms.

Args:

- `start_times` *(list[float])* — List of start times in seconds from the beginning of the recording for each spectrogram.
- `spec_duration` *(Optional[float])* — Length of each spectrogram in seconds. Defaults to cfg.audio.spec_duration.
- `freq_scale` *(Optional[str])* — Frequency scale to use ('linear', 'log', 'mel'). Defaults to cfg.audio.freq_scale.
- `decibels` *(Optional[float])* — Whether to convert to decibels. Defaults to cfg.audio.decibels.
- `top_db` *(Optional[int])* — Maximum decibel value for normalization. Defaults to cfg.audio.top_db.
- `db_power` *(Optional[int])* — Power to apply after decibel conversion. Defaults to cfg.audio.db_power.

Returns:

- `tuple` *((normalized_specs, unnormalized_specs) where:)* — - normalized_specs: Spectrograms normalized to 0-1 range (torch.Tensor) - unnormalized_specs: Original spectrograms without normalization (torch.Tensor)

Note:
    Returns (None, None) if no audio signal is loaded.

**load**  
```python
Audio.load(self, path)
```
Load audio from the given recording file.

Loads an audio file and stores it in the Audio object for subsequent
spectrogram generation. Supports both mono and stereo recordings.
For stereo recordings, can automatically choose the cleaner channel
if choose_channel is enabled in the configuration.

Args:

- `path` *(str)* — Path to the audio recording file.

Returns:

- `tuple` *((signal, sampling_rate) where:)* — - signal: The loaded audio signal as numpy array - sampling_rate: The sampling rate (should equal cfg.audio.sampling_rate)

Note:
    If loading fails, signal will be None and an error will be logged.

**seconds**  
```python
Audio.seconds(self)
```
Get the duration of the loaded audio signal in seconds.

Returns:

- `float` *(Duration of the signal in seconds, or 0 if no signal is loaded.)*



**set_config**  
```python
Audio.set_config(self, cfg: Optional[britekit.core.base_config.BaseConfig] = None)
```
Set or update the audio configuration for spectrogram generation.

This method configures the audio processing parameters including sampling rate,
window length, frequency scales, and transforms. It should be called before
generating spectrograms to ensure proper configuration.

The main bottleneck in audio processing is the load function. We often want to
generate spectrograms of different forms from the same audio, so it's best to
only load it once. This can be accomplished by calling set_config to update
spectrogram config before calling get_spectrograms with new settings.

If max_frequency is changing, it is best to start with the highest frequency,
so we downsample rather than upsampling.

Args:

- `cfg` *(Optional[BaseConfig])* — Configuration object. If None, uses default config.



**signal_len**  
```python
Audio.signal_len(self)
```
Get the length of the loaded audio signal in samples.

Returns:

- `int` *(Number of samples in the signal, or 0 if no signal is loaded.)*



### Extractor
**Class**  
```python
Extractor(db: britekit.training_db.training_db.TrainingDatabase, class_name: str, class_code: Optional[str] = None, cat_name: Optional[str] = None, src_name: Optional[str] = None, overlap: Optional[float] = None, spec_group: Optional[str] = None)
```
Class for extracting spectrograms from recordings and inserting them into the database.

Attributes:
    db (TrainingDatabase): Training database
    class_name (str): Name of class
    class_code (str, optional): Class code, only used when new class is inserted into the database
    cat_name (str, optional): Category name used when class is inserted. Defaults to 'default'.
    src_name (str, optional): Source name used when recording is inserted. Defaults to 'default'.
    overlap (float, optional): Spectrogram overlap in seconds
    spec_group (str, optional): Spectrogram group name

**Public methods & properties**

**extract_all**  
```python
Extractor.extract_all(self, dir_path: str)
```
Extract spectrograms for all recordings in the given directory.

Args:

- `dir_path` *(str)* — Directory containing recordings.

Returns:

- Number of spectrograms inserted.



**extract_by_image**  
```python
Extractor.extract_by_image(self, rec_dir: str, spec_dir: str, dest_dir: Optional[str] = None)
```
Extract spectrograms that match names of spectrogram images in a given directory.
Typically the spectrograms were generated using the 'search' or 'plot-db' commands.

Args:

- `rec_dir` *(str)* — Directory containing recordings.
- `spec_dir` *(str)* — Directory containing spectrogram images.
- `dest_dir` *(str, optional)* — Optionally copy used recordings to this directory.

Returns:

- Number of spectrograms inserted.



**insert_spectrograms**  
```python
Extractor.insert_spectrograms(self, recording_path, offsets)
```
Insert a spectrogram at each of the given offsets of the specified file.

Args:

- `recording_path` *(str)* — Path to audio recording.
- `offsets` *(list[float])* — List of offsets, where each represents number of seconds to start of spectrogram.

Returns:

- Number of spectrograms inserted.



### OccurrenceDataProvider
**Class**  
```python
OccurrenceDataProvider(db: britekit.occurrence_db.occurrence_db.OccurrenceDatabase)
```
Data access layer on top of OccurrenceDatabase.
If you insert or delete records after creating an instance of this,
you must call the refresh method.

Args:

- `db` *(OccurrenceDatabase)* — The database object.

**Public methods & properties**

**average_occurrences**  
```python
OccurrenceDataProvider.average_occurrences(self, county_prefix: str, class_name: str, area_weight: bool = False)
```
Given a county code prefix and class name, return the average occurrence values.
This is used for regional groupings,e.g. county_prefix = "CA" returns the average for Canada
and "CA-ON" returns the average for Ontario, when eBird county prefixes are used.
If area_weight = True, weight each county by its area.

Args:

- `county_prefix` *(str)* — County code prefix
- `class_name` *(str)* — Class name
- `area_weight` *(bool, Optional)* — If true, weight by county area (default = False)

Returns:

- Numpy array of 48 average occurrence values (one per week, using 4-week months).



**find_county**  
```python
OccurrenceDataProvider.find_county(self, latitude: float, longitude: float)
```
Return county info for a given latitude/longitude, or None if not found.

Args:

- `latitude` *(float)* — Latitude.
- `longitude` *(float)* — Longitude.

Returns:

- County object, or None if not found.



**max_occurrence**  
```python
OccurrenceDataProvider.max_occurrence(self, county_prefix: str, class_name: str, area_weight: bool = False)
```
Given a county code prefix and class name, return the average maximum occurrence value.
This is used for regional groupings,e.g. county_prefix = "CA" returns the average for Canada
and "CA-ON" returns the average for Ontario, when eBird county prefixes are used.
If area_weight = True, weight each county by its area.

This is not quite the same as average_occurrences.max(), since maximum values in each
county don't occur in the same week.

Args:

- `county_prefix` *(str)* — County code prefix
- `class_name` *(str)* — Class name
- `area_weight` *(bool, Optional)* — If true, weight by county area (default = False)

Returns:

- Numpy average maximum occurrence value.



**occurrences**  
```python
OccurrenceDataProvider.occurrences(self, county_code: str, class_name: str)
```
Return list of occurrence values for given county code and class name.

Args:

- `county_code` *(str)* — County code
- `class_name` *(str)* — Class name

Returns:

- List of occurrence values.



**refresh**  
```python
OccurrenceDataProvider.refresh(self)
```
Cache database info for quick access

**smoothed_occurrences**  
```python
OccurrenceDataProvider.smoothed_occurrences(self, county_code: str, class_name: str)
```
Return list of occurrence values for given county code and class name.
For each week, return the maximum of it and the adjacent weeks.

Args:

- `county_code` *(str)* — County code
- `class_name` *(str)* — Class name

Returns:

- List of smoothed occurrence values.



### OccurrenceDatabase
**Class**  
```python
OccurrenceDatabase(db_path: str = 'data/occurrence.db')
```
SQLite database interface for class occurrence data.

Attributes:
    db_path: Path to the database file.

**Public methods & properties**

**close**  
```python
OccurrenceDatabase.close(self)
```
Close the database.

**delete_class**  
```python
OccurrenceDatabase.delete_class(self, id)
```
Delete a class record specified by ID.

**delete_county**  
```python
OccurrenceDatabase.delete_county(self, id)
```
Delete a county record specified by ID.

**get_all_classes**  
```python
OccurrenceDatabase.get_all_classes(self)
```
Return a list of all classes.

**get_all_counties**  
```python
OccurrenceDatabase.get_all_counties(self)
```
Return a list of all counties.

**get_all_occurrences**  
```python
OccurrenceDatabase.get_all_occurrences(self)
```
Return a list with the CountyID and ClassID for every defined occurrence.

**get_occurrences**  
```python
OccurrenceDatabase.get_occurrences(self, county_id, class_name)
```
Return the occurrence values for a given county ID and class name.

**insert_class**  
```python
OccurrenceDatabase.insert_class(self, name)
```
Insert a class record and return the ID.

**insert_county**  
```python
OccurrenceDatabase.insert_county(self, name, code, min_x, max_x, min_y, max_y)
```
Insert a county record and return the ID.

**insert_occurrences**  
```python
OccurrenceDatabase.insert_occurrences(self, county_id, class_id, value)
```
Insert an occurrence record for a given county and class.

### PerMinuteTester
**Class**  
```python
PerMinuteTester(annotation_path: str, recording_dir: str, label_dir: str, output_dir: str, threshold: float, gen_pr_table: bool = False)
```
Calculate test metrics when annotations are specified per minute. That is, for selected minutes of
each recording, a list of classes known to be present is given, and we are to calculate metrics for
those minutes only.

Annotations are read as a CSV with three columns: "recording", "minute", and "classes".
The recording column is the file name without the path or type suffix, e.g. "recording1".
The minute column contains 1 for the first minute, 2 for the second minute etc. and may
exclude some minutes. The classes column contains a comma-separated list of codes for the classes found in the corresponding minute.
If your annotations are in a different format, simply convert to this format to use this script.

Classifiers should be run with a threshold of 0, and with label merging disabled so segment-specific scores are retained.

Attributes:
    annotation_path (str): Annotations CSV file.
    recording_dir (str): Directory containing recordings.
    label_dir (str): Directory containing Audacity labels.
    output_dir (str): Output directory, where reports will be written.
    threshold (float): Score threshold for precision/recall reporting.
    gen_pr_table (bool, optional): If true, generate a PR table, which may be slow (default = False).

**Public methods & properties**

**get_annotations**  
```python
PerMinuteTester.get_annotations(self)
```
Load annotation data from CSV file and process into internal format.

This method reads a CSV file containing ground truth annotations where each row
represents a recording, minute, and its associated classes. The CSV should have columns:
"recording" (filename without path/extension), "minute" (minute number starting from 1),
and "classes" (comma-separated class codes).

The method processes the annotations, handles class code mapping, filters out
unknown classes, and organizes the data for subsequent analysis.

Note:
    Sets self.annotations, self.annotated_class_set, self.annotated_classes,
    and self.segments_per_recording. Calls self.set_class_indexes() to update class indexing.

**get_pr_table**  
```python
PerMinuteTester.get_pr_table(self)
```
Calculate precision-recall table across multiple thresholds.

This method evaluates precision and recall metrics at different threshold values
(0.01 to 1.00 in 0.01 increments) to create comprehensive precision-recall curves.
It calculates both per-minute granularity metrics and per-second granularity metrics.

Returns:

- `dict` *(Dictionary containing precision-recall data with keys:)* — - annotated_thresholds: List of threshold values for annotated classes - annotated_precisions_minutes: List of precision values (minutes) for annotated classes - annotated_precisions_seconds: List of precision values (seconds) for annotated classes - annotated_recalls: List of recall values for annotated classes - trained_thresholds: List of threshold values for trained classes - trained_precisions: List of precision values for trained classes - trained_recalls: List of recall values for trained classes - annotated_thresholds_fine: Fine-grained thresholds for annotated classes - annotated_precisions_fine: Fine-grained precision values for annotated classes - annotated_recalls_fine: Fine-grained recall values for annotated classes - trained_thresholds_fine: Fine-grained thresholds for trained classes - trained_precisions_fine: Fine-grained precision values for trained classes - trained_recalls_fine: Fine-grained recall values for trained classes

Note:
    Uses both manual threshold evaluation and scikit-learn's precision_recall_curve
    for comprehensive coverage.

**produce_reports**  
```python
PerMinuteTester.produce_reports(self)
```
Generate comprehensive output reports and CSV files.

This method creates multiple output files containing detailed analysis results:
- Precision-recall tables and curves (if gen_pr_table=True)
- ROC-AUC curves and analysis
- Summary report with key metrics
- Recording-level details and summaries
- Class-level performance statistics
- Prediction range distribution analysis

The method generates the following files in the output directory:
- pr_per_threshold_*.csv/png: Precision-recall data at different thresholds
- pr_curve_*.csv/png: Precision-recall curves
- roc_*.csv/png: ROC-AUC curve analysis
- summary_report.txt: Human-readable summary with key metrics
- recording_details_trained.csv: Detailed statistics per recording/segment
- recording_summary_trained.csv: Summary statistics per recording
- class_annotated.csv: Performance metrics per annotated class
- class_non_annotated.csv: Prediction statistics for non-annotated classes
- prediction_range_counts.csv: Distribution of prediction scores

Note:
    Requires that self.map_dict, self.roc_dict, self.details_dict, and
    self.pr_table_dict (if gen_pr_table=True) have been calculated by calling
    the corresponding methods.

**run**  
```python
PerMinuteTester.run(self)
```
Execute the complete testing workflow.

This method orchestrates the entire testing process by:
1. Initializing the tester and loading data
2. Calculating PR-AUC (Precision-Recall Area-Under-Curve) statistics
3. Calculating ROC-AUC (Receiver Operating Characteristic Area-Under-Curve) statistics
4. Calculating precision-recall statistics at the specified threshold
5. Generating a precision-recall table across multiple thresholds (if gen_pr_table=True)
6. Producing comprehensive output reports

The method calls all necessary setup, calculation, and reporting methods
in the correct order to complete the analysis workflow.

Note:
    This is the main entry point for running a complete test analysis.
    All output files will be written to self.output_dir.
    The gen_pr_table parameter controls whether detailed PR table analysis is performed.

### PerRecordingTester
**Class**  
```python
PerRecordingTester(annotation_path: str, recording_dir: str, label_dir: str, output_dir: str, threshold: float, tp_secs_at_precision: float = 0.95)
```
Calculate test metrics when annotations are specified per recording. That is, the ground truth data
gives a list of classes per recording, with no indication of where in the recording they are heard.
This has the advantage that new tests can be created very quickly. By assuming that all detections
of a valid class are valid, we can count the number of TP and FP seconds. However, FNs can only be
counted at the recording level, so our recall measure is extremely coarse. To work around this, we can
output the number of TP seconds at a given precision, say 95%. Given two tests, this can be used to
measure relative but not absolute recall.

Annotations are defined in a CSV with two columns: "recording", and "classes".
The recording column is the file name without the path or type suffix, e.g. "recording1".
The classes column contains a comma-separated list of codes for the classes found in the corresponding
recording. If your annotations are in a different format, simply convert to this format to use this script.

Classifiers should be run with a threshold of 0, and with label merging disabled so segment-specific scores are retained.

Attributes:
    annotation_path (str): Annotations CSV file.
    recording_dir (str): Directory containing recordings.
    label_dir (str): Directory containing Audacity labels.
    output_dir (str): Output directory, where reports will be written.
    threshold (float): Score threshold for precision/recall reporting.
    tp_secs_at_precision (float, optional): Granular recall cannot be calculated with per-recording annotations,
    but reporting TP seconds at this precision is a useful proxy (default=.95).

**Public methods & properties**

**get_annotations**  
```python
PerRecordingTester.get_annotations(self)
```
Load annotation data from CSV file and process into internal format.

This method reads a CSV file containing ground truth annotations where each row
represents a recording and its associated classes. The CSV should have columns:
"recording" (filename without path/extension) and "classes" (comma-separated class codes).

The method processes the annotations, handles class code mapping, filters out
unknown classes, and organizes the data for subsequent analysis.

Note:
    Sets self.annotations, self.annotated_class_set, and self.annotated_classes.
    Calls self.set_class_indexes() to update class indexing.

**get_pr_table**  
```python
PerRecordingTester.get_pr_table(self)
```
Calculate precision-recall table across multiple thresholds.

This method evaluates precision and recall metrics at different threshold values
(0.01 to 1.00 in 0.01 increments) to create a comprehensive precision-recall curve.
It calculates both per-recording granularity metrics and per-second granularity metrics.

Returns:

- `dict` *(Dictionary containing precision-recall data with keys:)* — - precisions: List of precision values at each threshold - recalls: List of recall values at each threshold - precision_secs: List of precision values in seconds at each threshold - tp_secs: List of true positive seconds at each threshold - fp_secs: List of false positive seconds at each threshold - thresholds: List of threshold values used

Note:
    Rows with precision=0 at the end are trimmed to avoid unnecessary data points.

**produce_reports**  
```python
PerRecordingTester.produce_reports(self)
```
Generate comprehensive output reports and CSV files.

This method creates multiple output files containing detailed analysis results:
- Precision-recall table and curve data
- Summary report with key metrics
- Recording-level details and summaries
- Class-level performance statistics

The method generates the following files in the output directory:
- pr_table.csv: Precision-recall data at different thresholds
- pr_curve.csv: Interpolated precision-recall curve
- summary_report.txt: Human-readable summary with key metrics
- recording_details.csv: Detailed statistics per recording
- recording_summary.csv: Summary statistics per recording
- class.csv: Performance metrics per class

Note:
    Requires that self.map_dict, self.roc_dict, self.details_dict, and
    self.pr_table_dict have been calculated by calling the corresponding methods.

**run**  
```python
PerRecordingTester.run(self)
```
Execute the complete testing workflow.

This method orchestrates the entire testing process by:
1. Initializing the tester and loading data
2. Calculating PR-AUC (Precision-Recall Area-Under-Curve) statistics
3. Calculating ROC-AUC (Receiver Operating Characteristic Area-Under-Curve) statistics
4. Calculating precision-recall statistics at the specified threshold
5. Generating a precision-recall table across multiple thresholds
6. Producing comprehensive output reports

The method calls all necessary setup, calculation, and reporting methods
in the correct order to complete the analysis workflow.

Note:
    This is the main entry point for running a complete test analysis.
    All output files will be written to self.output_dir.

### PerSegmentTester
**Class**  
```python
PerSegmentTester(annotation_path: str, recording_dir: str, label_dir: str, output_dir: str, threshold: float, calibrate: bool = False, cutoff: float = 0.6, coef: Optional[float] = None, inter: Optional[float] = None)
```
Calculate test metrics when individual sounds are annotated in the ground truth data.
Annotations are read as a CSV with four columns: recording, class, start_time and end_time.
The recording column is the file name without the path or type suffix, e.g. "recording1".
The class column contains the class code, and start_time and end_time are
fractional seconds, e.g. 12.5 represents 12.5 seconds from the start of the recording.
If your annotations are in a different format, simply convert to this format to use this script.

Classifiers should be run with a threshold of 0, and with label merging disabled so segment-specific scores are retained.

Attributes:
    annotation_path (str): Annotations CSV file.
    recording_dir (str): Directory containing recordings.
    label_dir (str): Directory containing Audacity labels.
    output_dir (str): Output directory, where reports will be written.
    threshold (float): Score threshold for precision/recall reporting.
    trained_classes (list): List of trained class codes.

**Public methods & properties**

**do_calibration**  
```python
PerSegmentTester.do_calibration(self)
```
Calculate and print optimal Platt scaling coefficient and intercept.

This method performs Platt scaling calibration on the model predictions to improve
probability calibration. It uses logistic regression to find optimal scaling parameters
that transform the raw logits to better-calibrated probabilities.

The method filters predictions above the calibration cutoff threshold, fits a
logistic regression model to the logits, and outputs the optimal coefficient and
intercept values. It also generates a calibration curve plot for visualization.

Note:
    Requires that self.y_pred_trained and self.y_true_trained have been initialized.
    The calibration_cutoff parameter controls which predictions are used for fitting.
    Generates a calibration plot saved to the output directory.

Raises:

- `ValueError` *(If too few samples are above the calibration cutoff threshold.)*



**get_annotations**  
```python
PerSegmentTester.get_annotations(self)
```
Load annotation data from CSV file and process into internal format.

This method reads a CSV file containing ground truth annotations where each row
represents a recording, class, start time, and end time. The CSV should have columns:
"recording" (filename without path/extension), "class" (class code),
"start_time" (fractional seconds from start), and "end_time" (fractional seconds from start).

The method processes the annotations, handles class code mapping, filters out
unknown classes, and organizes the data into Annotation objects for subsequent analysis.

Note:
    Sets self.annotations, self.annotated_class_set, and self.annotated_classes.
    Calls self.set_class_indexes() to update class indexing.

**get_offsets**  
```python
PerSegmentTester.get_offsets(start_time, end_time, segment_len, overlap, min_seconds=0.3)
```
Determine which offsets an annotation or label should be assigned to.

This static method calculates the time offsets where an annotation or label should
be assigned based on the segment boundaries. The returned offsets are aligned on
boundaries of segment_len - overlap.

Args:

- `start_time` — Start time of the annotation in seconds
- `end_time` — End time of the annotation in seconds
- `segment_len` — Length of each segment in seconds
- `overlap` — Overlap between consecutive segments in seconds
- `min_seconds` — Minimum number of seconds that must be contained in the first and last segments (default: 0.3)

Returns:

- `list` *(List of time offsets where the annotation should be assigned)*

Note:
    For example, if segment_len=3 and overlap=1.5, segments are aligned on
    1.5 second boundaries (0, 1.5, 3.0, ...). The method ensures that the
    first and last segments contain at least min_seconds of the labelled sound.

**get_pr_table**  
```python
PerSegmentTester.get_pr_table(self)
```
Calculate precision-recall table across multiple thresholds.

This method evaluates precision and recall metrics at different threshold values
(0.01 to 1.00 in 0.01 increments) to create comprehensive precision-recall curves.
It calculates both per-minute granularity metrics and per-second granularity metrics.

Returns:

- `dict` *(Dictionary containing precision-recall data with keys:)* — - annotated_thresholds: List of threshold values for annotated classes - annotated_precisions_minutes: List of precision values (minutes) for annotated classes - annotated_precisions_seconds: List of precision values (seconds) for annotated classes - annotated_recalls: List of recall values for annotated classes - trained_thresholds: List of threshold values for trained classes - trained_precisions: List of precision values for trained classes - trained_recalls: List of recall values for trained classes - annotated_thresholds_fine: Fine-grained thresholds for annotated classes - annotated_precisions_fine: Fine-grained precision values for annotated classes - annotated_recalls_fine: Fine-grained recall values for annotated classes - trained_thresholds_fine: Fine-grained thresholds for trained classes - trained_precisions_fine: Fine-grained precision values for trained classes - trained_recalls_fine: Fine-grained recall values for trained classes

Note:
    Uses both manual threshold evaluation and scikit-learn's precision_recall_curve
    for comprehensive coverage.

**get_segments**  
```python
PerSegmentTester.get_segments(self, start_time, end_time, min_seconds=0.3)
```
Convert time offsets to segment indexes.

This method converts the time offsets returned by get_offsets() into segment
indexes that correspond to the actual segments in the analysis.

Args:

- `start_time` — Start time of the annotation in seconds
- `end_time` — End time of the annotation in seconds
- `min_seconds` — Minimum number of seconds that must be contained in segments (default: 0.3)

Returns:

- `list` *(List of segment indexes where the annotation should be assigned)*

Note:
    Uses self.segment_len and self.overlap to calculate segment boundaries.
    Returns an empty list if no valid segments are found.

**initialize**  
```python
PerSegmentTester.initialize(self)
```
Initialize

**plot_calibration_curve**  
```python
PerSegmentTester.plot_calibration_curve(self, y_true, y_pred, a, b, n_bins=15)
```
Plot calibration curve comparing uncalibrated and Platt-calibrated predictions.

This method creates a reliability diagram (calibration curve) that shows how well
the model's predicted probabilities match the observed frequencies. It plots both
the original uncalibrated predictions and the Platt-scaled calibrated predictions
against the ideal calibration line.

Args:

- `y_true` — Ground truth labels (0 or 1)
- `y_pred` — Uncalibrated model probabilities
- `a` — Platt scaling coefficient
- `b` — Platt scaling intercept
- `n_bins` — Number of bins for the calibration curve (default: 15)
- `Note` — Saves the calibration plot to the output directory with filename format: calibration-{a:.2f}-{b:.2f}.png

Note:
    Saves the calibration plot to the output directory with filename format:
    calibration-{a:.2f}-{b:.2f}.png

**run**  
```python
PerSegmentTester.run(self)
```
Execute the complete testing workflow.

This method orchestrates the entire testing process by:
1. Initializing the tester and loading data
2. If calibrate=True, performing calibration analysis and returning early
3. Calculating PR-AUC (Precision-Recall Area-Under-Curve) statistics
4. Calculating ROC-AUC (Receiver Operating Characteristic Area-Under-Curve) statistics
5. Calculating precision-recall statistics at the specified threshold
6. Generating a precision-recall table across multiple thresholds
7. Producing comprehensive output reports

The method calls all necessary setup, calculation, and reporting methods
in the correct order to complete the analysis workflow.

Note:
    This is the main entry point for running a complete test analysis.
    If calibrate=True, only calibration analysis is performed and the method returns early.
    All output files will be written to self.output_dir.

### Pickler
**Class**  
```python
Pickler(db_path: str, output_path: str, classes_path: Optional[str] = None, max_per_class: Optional[int] = None, spec_group: Optional[str] = None)
```
Create a pickle file from selected training records, for input to training.

Attributes:
    db_path (str): path to database.
    output_path (str): output_path.
    classes_path (Optional, str): path to CSV file listing classes.
    max_per_class (int, optional): maximum spectrograms to output per class.

**Public methods & properties**

**pickle**  
```python
Pickler.pickle(self, quiet=False)
```
Create the pickle file as specified.

### Predictor
**Class**  
```python
Predictor(model_path: str, device: Optional[str] = None)
```
Given a recording and a model or ensemble of models, provide methods to return scores in several formats.

**Public methods & properties**

**get_dataframe**  
```python
Predictor.get_dataframe(self, score_array, frame_map, start_times: list[float], recording_name: str)
```
Given an array of raw scores, return as a pandas dataframe.

Args:

- `score_array` *(np.ndarray)* — Array of scores of shape (num_spectrograms, num_species).
- `frame_map` *(np.ndarray, optional)* — Frame-level scores of shape (num_frames, num_species). If provided, uses frame-level labels; otherwise uses segment-level labels.
- `start_times` *(list[float])* — Start time in seconds for each spectrogram.
- `recording_name` *(str)* — Name of the recording for the dataframe.

Returns:

- pd.DataFrame: DataFrame with columns ['recording', 'name', 'start_time', 'end_time', 'score']
- containing all detected species segments.



**get_frame_labels**  
```python
Predictor.get_frame_labels(self, frame_map) -> dict[str, list[britekit.core.predictor.Label]]
```
Given a frame map, return dict of labels.

Args:

- `frame_map` *(np.ndarray)* — Array of scores of shape (num_frames, num_species).

Returns:

- dict[str, list]: Dictionary mapping species names to lists of Label objects.
- Each Label contains (score, start_time, end_time) for detected segments.
- Labels are either variable-duration (if segment_len is None) or
- fixed-duration based on cfg.infer.segment_len.



**get_raw_scores**  
```python
Predictor.get_raw_scores(self, recording_path: str)
```
Get scores in array format from the loaded models for the given recording.

Args:

- `recording_path` *(str)* — Path to the audio recording file.

Returns:

- `tuple` *(A tuple containing:)* — - avg_score (np.ndarray): Average scores across all models in the ensemble. Shape is (num_spectrograms, num_classes). - avg_frame_map (np.ndarray, optional): Average frame-level scores if using SED models. Shape is (num_frames, num_classes). None if not using SED models. - start_times (list[float]): Start time in seconds for each spectrogram.



**get_segment_labels**  
```python
Predictor.get_segment_labels(self, scores, start_times: list[float]) -> dict[str, list[britekit.core.predictor.Label]]
```
Given an array of raw segment-level scores, return dict of labels.

Args:

- `scores` *(np.ndarray)* — Array of scores of shape (num_spectrograms, num_species).
- `start_times` *(list[float])* — Start time in seconds for each spectrogram.

Returns:

- dict[str, list]: Dictionary mapping species names to lists of Label objects.
- Each Label contains (score, start_time, end_time) for detected segments.



**save_audacity_labels**  
```python
Predictor.save_audacity_labels(self, scores, frame_map, start_times: list[float], file_path: str) -> None
```
Given an array of raw scores, convert to Audacity labels and save in the given file.

Args:

- `scores` *(np.ndarray)* — Segment-level scores of shape (num_spectrograms, num_species).
- `frame_map` *(np.ndarray, optional)* — Frame-level scores of shape (num_frames, num_species). If provided, uses frame-level labels; otherwise uses segment-level labels.
- `start_times` *(list[float])* — Start time in seconds for each spectrogram.
- `file_path` *(str)* — Output path for the Audacity label file.

Returns:

- `None` *(Writes the labels directly to the specified file.)*



**to_global_frames**  
```python
Predictor.to_global_frames(self, frame_scores, offsets_sec: Sequence[float], recording_duration_sec: float)
```
Map overlapping per-spectrogram frame scores onto a global frame grid.
Use mean rather than max or weighted values.

Args:

- `frame_scores` — (num_specs, num_classes, T_spec) scores in [0, 1].
- `offsets_sec` — start time (s) for each spectrogram within the recording.
- `recording_duration_sec` — total recording length in seconds.

Returns:

- `global_frames` *((num_classes, T_global) tensor of scores in [0, 1].)*



### Trainer
**Class**  
```python
Trainer()
```
Run training as specified in configuration.

**Public methods & properties**

**find_lr**  
```python
Trainer.find_lr(self, num_batches: int = 100)
```
Suggest a learning rate and produce a plot.

**run**  
```python
Trainer.run(self)
```
Run training as specified in configuration.

### TrainingDataProvider
**Class**  
```python
TrainingDataProvider(db: britekit.training_db.training_db.TrainingDatabase)
```
Data access layer on top of TrainingDatabase.

Args:

- `db` *(TrainingDatabase)* — The database object.

**Public methods & properties**

**category_id**  
```python
TrainingDataProvider.category_id(self, name)
```
Return the ID of the specified Category record (insert it if missing).

**class_id**  
```python
TrainingDataProvider.class_id(self, name, code, category_id)
```
Return the ID of the specified Class record (insert it if missing).

**class_info**  
```python
TrainingDataProvider.class_info(self)
```
Get a summary and details dataframe with class, recording and segment counts.

Returns:

- `summary_df` *(A pandas dataframe with recording and segment counts per class)* — details_df: A pandas dataframe with segment counts per recording per class



**recording_id**  
```python
TrainingDataProvider.recording_id(self, class_name, filename, path, source_id, seconds)
```
Return the ID of the specified Recording record (insert it if missing).

**segment_class_dict**  
```python
TrainingDataProvider.segment_class_dict(self)
```
Get a dict from segment ID to a set of class names.

Returns:

- dict from segment ID to a set of class names



**source_id**  
```python
TrainingDataProvider.source_id(self, filename, source_name=None)
```
Return the ID of the specified Source record (insert it if missing).

**spec_group_info**  
```python
TrainingDataProvider.spec_group_info(self)
```
Get a dataframe with number of spectrograms per spec group.

Returns:

- A pandas dataframe with number of spectrograms per spec group.



**specgroup_id**  
```python
TrainingDataProvider.specgroup_id(self, name)
```
Return the ID of the specified SpecGroup record (insert it if missing).

### TrainingDatabase
**Class**  
```python
TrainingDatabase(db_path: str = 'data/training.db')
```
Handle the creation, querying, and updating of a simple SQLite database storing
training data, including a class table, recording table and spectrogram tables.

Attributes:
    db_path: Path to the database file.

**Public methods & properties**

**close**  
```python
TrainingDatabase.close(self)
```
Close the database.

**delete_category**  
```python
TrainingDatabase.delete_category(self, filters: Optional[dict] = None)
```
Delete one or more Category records.

Args:

- `filters` *(dict, optional)* — a dict of column_name/value pairs that define filters. Valid column names for the Category table are: - ID (int): record ID - Name (str): source name

Returns:

- Number of records deleted.



**delete_class**  
```python
TrainingDatabase.delete_class(self, filters: Optional[dict] = None)
```
Delete one ore more Class records.

Args:

- `filters` *(dict, optional)* — a dict of column_name/value pairs that define filters.

Returns:

- Number of records deleted.



**delete_recording**  
```python
TrainingDatabase.delete_recording(self, filters: Optional[dict] = None)
```
Delete one or more Recording records.

Args:

- `filters` *(dict, optional)* — a dict of column_name/value pairs that define filters.

Returns:

- Number of records deleted.



**delete_segment**  
```python
TrainingDatabase.delete_segment(self, filters: Optional[dict] = None)
```
Delete one or more Segment records.

Args:

- `filters` *(dict, optional)* — a dict of column_name/value pairs that define filters.



**delete_soundtype**  
```python
TrainingDatabase.delete_soundtype(self, filters: Optional[dict] = None)
```
Delete one or more SoundType records.

Args:

- `filters` *(dict, optional)* — a dict of column_name/value pairs that define filters.

Returns:

- Number of records deleted.



**delete_source**  
```python
TrainingDatabase.delete_source(self, filters: Optional[dict] = None)
```
Delete one or more Source records.

Args:

- `filters` *(dict, optional)* — a dict of column_name/value pairs that define filters. Valid column names for the Source table are: - ID (int): record ID - Name (str): source name

Returns:

- Number of records deleted.



**delete_specgroup**  
```python
TrainingDatabase.delete_specgroup(self, filters: Optional[dict] = None)
```
Delete one or more SpecGroup records.

Args:

- `filters` *(dict, optional)* — a dict of column_name/value pairs that define filters. Valid column names for the Category table are: - ID (int): record ID - Name (str): specgroup name

Returns:

- Number of records deleted.



**delete_specvalue**  
```python
TrainingDatabase.delete_specvalue(self, filters: Optional[dict] = None)
```
Delete one or more SpecValue records.

Args:

- `filters` *(dict, optional)* — a dict of column_name/value pairs that define filters. Valid column names for the SpecValue table are: - ID (int): record ID

Returns:

- Number of records deleted.



**get_all_segment_counts**  
```python
TrainingDatabase.get_all_segment_counts(self)
```
Get the class name and segment count for all classes.

Args:
    None

Returns:

- A list of entries, each as a SimpleNamespace object with the following attributes:
- - class_name (str): Class Name.
- - count (int): Number of segments.



**get_category**  
```python
TrainingDatabase.get_category(self, filters: Optional[dict] = None)
```
Query the Category table.

Args:

- `filters` *(dict, optional)* — a dict of column_name/value pairs that define filters. Valid column names for the Category table are: - ID (int): record ID - Name (str): category name

Returns:

- A list of entries, each as a SimpleNamespace object with the following attributes:
- - id (int): Unique ID of the entry.
- - name (str): Name of the category.



**get_category_count**  
```python
TrainingDatabase.get_category_count(self, filters: Optional[dict] = None)
```
Get the number of records in the Category table.

Args:

- `filters` *(dict, optional)* — a dict of column_name/value pairs that define filters. Valid column names for the Category table are: - ID (int): record ID - Name (str): category name

Returns:

- Number of records that match the criteria.



**get_class**  
```python
TrainingDatabase.get_class(self, filters: Optional[dict] = None)
```
Query the Class table.

Args:

- `filters` *(dict, optional)* — a dict of column_name/value pairs that define filters.

Returns:

- A list of entries, each as a SimpleNamespace object with the following attributes:
- - id (int): Unique ID of the entry.
- - category_id (int): ID of the corresponding category.
- - name (str): Class name
- - alt_name (str): Class alt_name
- - code (str): Class code



**get_class_count**  
```python
TrainingDatabase.get_class_count(self, filters: Optional[dict] = None)
```
Get the number of records in the Class table.

Args:

- `filters` *(dict, optional)* — a dict of column_name/value pairs that define filters.

Returns:

- Number of records that match the criteria.



**get_recording**  
```python
TrainingDatabase.get_recording(self, filters: Optional[dict] = None)
```
Query the Recording table.

Args:

- `filters` *(dict, optional)* — a dict of column_name/value pairs that define filters.

Returns:

- A list of entries, each as a SimpleNamespace object with the following attributes:
- - id (int): Unique ID of the entry.
- - source_id (int): ID of the corresponding source.
- - class_id (int): ID of the corresponding class.
- - filename (str): File name
- - path (str): Path
- - seconds (float): Duration in seconds



**get_recording_by_class**  
```python
TrainingDatabase.get_recording_by_class(self, class_name: str)
```
Return all recordings that have segments with the given class.

Args:

- `class_name` *(str)* — name of the class.

Returns:

- A list of entries, each as a SimpleNamespace object with the following attributes:
- - id (int): Unique ID of the entry.
- - source_id (int): ID of the corresponding source.
- - class_id (int): ID of the corresponding class.
- - filename (str): File name
- - path (str): Path
- - seconds (float): Duration in seconds



**get_recording_count**  
```python
TrainingDatabase.get_recording_count(self, filters: Optional[dict] = None)
```
Get the number of records in the Recording table.

Args:

- `filters` *(dict, optional)* — a dict of column_name/value pairs that define filters.

Returns:

- Number of records that match the criteria.



**get_segment**  
```python
TrainingDatabase.get_segment(self, filters: Optional[dict] = None, include_audio: bool = False)
```
Query the Segment table.

Args:

- `filters` *(dict, optional)* — a dict of column_name/value pairs that define filters.
- `include_audio` *(bool, optional)* — if True, include audio in the returned objects. Default = False.

Returns:

- A list of entries, each as a SimpleNamespace object with the following attributes:
- - id (int): Unique ID of the entry.
- - audio (blob): raw audio, or None.
- - sampling_rate (int): if there is audio, this is its sampling_rate
- - offset (float): number of seconds from the start of the recording to the start of the segment.
- - recording_id (int): ID of the corresponding Recording record.



**get_segment_by_class**  
```python
TrainingDatabase.get_segment_by_class(self, class_name: str, include_audio: bool = False)
```
Get segment info for the given class.

Args:

- `class_name` *(str)* — class name.
- `include_audio` *(bool, optional)* — if True, include audio in the returned objects. Default = False.

Returns:

- A list of entries, each as a SimpleNamespace object with the following attributes:
- - id (int): ID of the Segment record.
- - audio (blob): raw audio, or None.
- - recording_id (int): ID of the corresponding Recording record.
- - offset (float): Number of seconds from the start of the recording to the start of the segment.



**get_segment_class**  
```python
TrainingDatabase.get_segment_class(self)
```
Get all records from the SegmentClass table.

Returns:

- A list of entries, each as a SimpleNamespace object with the following attributes:
- - segment_id (int): Segment ID.
- - class_id (int): Class ID.
- - soundtype_id: SoundType ID.



**get_segment_class_count**  
```python
TrainingDatabase.get_segment_class_count(self, filters: Optional[dict] = None)
```
Get the number of records in the SegmentClass table.

Args:

- `filters` *(dict, optional)* — a dict of column_name/value pairs that define filters.

Returns:

- Number of records that match the criteria.



**get_segment_count**  
```python
TrainingDatabase.get_segment_count(self, filters: Optional[dict] = None)
```
Get the number of records in the Segment table.

Args:

- `filters` *(dict, optional)* — a dict of column_name/value pairs that define filters.

Returns:

- Number of records that match the criteria.



**get_soundtype**  
```python
TrainingDatabase.get_soundtype(self, filters: Optional[dict] = None)
```
Query the SoundType table.

Args:

- `filters` *(dict, optional)* — a dict of column_name/value pairs that define filters.

Returns:

- A list of entries, each as a SimpleNamespace object with the following attributes:
- - id (int): Unique ID of the entry.
- - source_id (int): ID of the corresponding source.
- - class_id (int): ID of the corresponding class.
- - filename (str): File name
- - path (str): Path
- - seconds (float): Duration in seconds



**get_soundtype_count**  
```python
TrainingDatabase.get_soundtype_count(self, filters: Optional[dict] = None)
```
Get the number of records in the SoundType table.

Args:

- `filters` *(dict, optional)* — a dict of column_name/value pairs that define filters.

Returns:

- Number of records that match the criteria.



**get_source**  
```python
TrainingDatabase.get_source(self, filters: Optional[dict] = None)
```
Query the Source table.

Args:

- `filters` *(dict, optional)* — a dict of column_name/value pairs that define filters. Valid column names for the Source table are: - ID (int): record ID - Name (str): source name

Returns:

- A list of entries, each as a SimpleNamespace object with the following attributes:
- - id (int): Unique ID of the entry.
- - name (str): Name of the source.



**get_source_count**  
```python
TrainingDatabase.get_source_count(self, filters: Optional[dict] = None)
```
Get the number of records in the Source table.

Args:

- `filters` *(dict, optional)* — a dict of column_name/value pairs that define filters. Valid column names for the Source table are: - ID (int): record ID - Name (str): source name

Returns:

- Number of records that match the criteria.



**get_specgroup**  
```python
TrainingDatabase.get_specgroup(self, filters: Optional[dict] = None)
```
Query the SpecGroup table.

Args:

- `filters` *(dict, optional)* — a dict of column_name/value pairs that define filters. Valid column names for the SpecGroup table are: - ID (int): record ID - Name (str): specgroup name

Returns:

- A list of entries, each as a SimpleNamespace object with the following attributes:
- - id (int): Unique ID of the entry.
- - name (str): Name of the specgroup.



**get_specgroup_count**  
```python
TrainingDatabase.get_specgroup_count(self, filters: Optional[dict] = None)
```
Get the number of records in the SpecGroup table.

Args:

- `filters` *(dict, optional)* — a dict of column_name/value pairs that define filters. Valid column names for the Source table are: - ID (int): record ID - Name (str): spec group name

Returns:

- Number of records that match the criteria.



**get_spectrogram_by_class**  
```python
TrainingDatabase.get_spectrogram_by_class(self, class_name: str, include_value: bool = True, include_embedding: bool = False, spec_group: str = 'default', limit: Optional[int] = None)
```
Fetch joined spectrogram records for the given class.

Args:

- `class_name` *(str)* — class name.
- `include_value` *(bool, optional)* — If True, include the compressed spectrogram. Default = True.
- `include_embedding` *(bool, optional)* — If True, include embeddings in the returned objects. Default = False.
- `spec_group` *(str)* — Name of spectrogram group. Default = "default".
- `limit` *(int, optional)* — If specified, only return up to this many records.

Returns:

- A list of entries, each as a SimpleNamespace object with the following attributes:
- - segment_id (int): ID of the Segment record.
- - specvalue_id (int): ID of the SpecValue record.
- - value (bytes): The spectrogram itself.
- - offset (float): Number of seconds from the start of the recording to the start of the segment.
- - recording_id (int): ID of the corresponding Recording record.



**get_specvalue**  
```python
TrainingDatabase.get_specvalue(self, filters: Optional[dict] = None)
```
Query the SpecValue table.

Args:

- `filters` *(dict, optional)* — a dict of column_name/value pairs that define filters.

Returns:

- A list of entries, each as a SimpleNamespace object with the following attributes:
- - id (int): Unique ID of the entry.
- - value (bytes): The compressed spectrogram.
- - embedding (bytes): The embedding.
- - specgroup_id (str): ID of the corresponding specgroup.
- - segment_id (str): ID of the corresponding segment.



**get_specvalue_count**  
```python
TrainingDatabase.get_specvalue_count(self, filters: Optional[dict] = None)
```
Get the number of records in the SpecValue table.

Args:

- `filters` *(dict, optional)* — a dict of column_name/value pairs that define filters.

Returns:

- Number of records that match the criteria.



**insert_category**  
```python
TrainingDatabase.insert_category(self, name: str)
```
Insert a Category record.

Args:

- `name` *(str)* — Name of the category (e.g. "bird").

Returns:

- row_id (int): ID of the inserted record.



**insert_class**  
```python
TrainingDatabase.insert_class(self, category_id: int, name: str, alt_name: Optional[str] = None, code: Optional[str] = None, alt_code: Optional[str] = None)
```
Insert a Class record.

Args:

- `category_id` *(int, required)* — Record ID of the category (e.g. ID of "bird" in the Category table).
- `name` *(str, required)* — Name of the class (e.g. "White-winged Crossbill").
- `alt_name` *(str, optional)* — Alternate name of the class (e.g. "Two-barred Crossbill").
- `code` *(str, optional)* — Code for the class (e.g. "WWCR").
- `alt_code` *(str, optional)* — Alternate code

Returns:

- row_id (int): ID of the inserted record.



**insert_recording**  
```python
TrainingDatabase.insert_recording(self, source_id: int, filename: str, path: str, seconds: float = 0)
```
Insert a Recording record.

Args:

- `source_id` *(int, required)* — Record ID of the source (e.g. ID of "Xeno-Canto" in the Source table).
- `filename` *(str, required)* — Name of the recording (e.g. "XC12345.mp3").
- `path` *(str, required)* — Full path to the recording.
- `seconds` *(float, optional)* — Duration of the recording in seconds.

Returns:

- row_id (int): ID of the inserted record.



**insert_segment**  
```python
TrainingDatabase.insert_segment(self, recording_id: int, offset: float)
```
Insert a Segment record.

Args:

- `recording_id` *(int, required)* — Record ID of the recording.
- `offset` *(float, required)* — offset in seconds from start of the recording.
- `audio` *(blob, optional)* — corresponding raw audio.

Returns:

- row_id (int): ID of the inserted record.



**insert_segment_class**  
```python
TrainingDatabase.insert_segment_class(self, segment_id: int, class_id: int)
```
Insert a SegmentClass record, to identify a segment as containing a sound of the class.
Spectrograms can contain sounds of multiple classes, represented by multiple SegmentClass
records.

Args:

- `segment_id` *(int, required)* — Segment ID.
- `class_id` *(int, required)* — Class ID.

Returns:

- row_id (int): ID of the inserted record.



**insert_soundtype**  
```python
TrainingDatabase.insert_soundtype(self, name: str)
```
Insert a SoundType record.

Args:

- `name` *(str, required)* — Name of the sound type.

Returns:

- row_id (int): ID of the inserted record.



**insert_source**  
```python
TrainingDatabase.insert_source(self, name: str)
```
Insert a Source record.

Args:

- `name` *(str)* — Name of the source (e.g. "Xeno-Canto").

Returns:

- row_id (int): ID of the inserted record.



**insert_specgroup**  
```python
TrainingDatabase.insert_specgroup(self, name: str)
```
Insert a SpecGroup record.

Args:

- `name` *(str)* — Name of the group (e.g. "logscale").

Returns:

- row_id (int): ID of the inserted record.



**insert_specvalue**  
```python
TrainingDatabase.insert_specvalue(self, value: bytes, spec_group_id: int, segment_id: int)
```
Insert a SpecValue record.

Args:

- `value` *(blob, required)* — the actual compressed spectrogram
- `spec_group_id` *(int, required)* — ID of spec group record
- `segment_id` *(int, required)* — ID of segment record
- `sampling_rate` *(int)* — sampling rate used to create it

Returns:

- row_id (int): ID of the inserted record.



**update_recording**  
```python
TrainingDatabase.update_recording(self, id: int, field: str, value)
```
Update a record in the Recording table.

Args:

- `id` *(int)* — ID that identifies the record to update
- `field` *(str)* — Name of column to update.
- `value` — New value.



**update_segment**  
```python
TrainingDatabase.update_segment(self, id: int, field: str, value)
```
Update a record in the Segment table.

Args:

- `id` *(int)* — ID that identifies the record to update
- `field` *(str)* — Name of column to update.
- `value` — New value.



**update_segment_class**  
```python
TrainingDatabase.update_segment_class(self, id: int, field: str, value)
```
Update a record in the SegmentClass table.

Args:

- `id` *(int)* — ID that identifies the record to update
- `field` *(str)* — Name of column to update.
- `value` — New value.



**update_specvalue**  
```python
TrainingDatabase.update_specvalue(self, id: int, field: str, value)
```
Update a record in the SpecValue table.

Args:

- `id` *(int)* — ID that identifies the record to update
- `field` *(str)* — Name of column to update.
- `value` — New value.



### Tuner
**Class**  
```python
Tuner(recording_dir: str, output_dir: str, annotation_path: str, train_log_dir: str, metric: str, param_space: Optional[Any], num_trials: int = 0, num_runs: int = 1, extract: bool = False, skip_training: bool = False, classes_path: Optional[str] = None)
```
Tune the joint values of selected hyperparameters, either by exhaustive or random search.

**Public methods & properties**

**run**  
```python
Tuner.run(self)
```
Initiate the search and return the best score and best hyperparameter values.
A "trial" is a set of parameter values and a "run" is a training/inference run.
There may be multiple runs per trial, since results per run are non-deterministic.

### __version__
str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### get_config
**Function**  
```python
get_config(cfg_path: Optional[str] = None) -> britekit.core.base_config.BaseConfig
```
