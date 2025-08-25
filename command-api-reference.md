## Command API Reference
### add_cat
**Function**  
```python
add_cat_cmd(db_path, name)
```
Add a category (class group) record to the training database.

Categories are used to group related classes together in the database.
For example, you might have categories like "Birds", "Mammals", or "Insects"
that contain multiple related species classes.

Args:

- `db_path` *(str, optional)* — Path to the training database. Defaults to cfg.train.train_db.
- `name` *(str)* — Name of the category to add (e.g., "Birds", "Mammals").



### add_class
**Function**  
```python
add_class_cmd(db_path, category, name, code, alt_name, alt_code)
```
Add a class record to the training database.

Classes represent the target species or sound categories for training and inference.
Each class must belong to a category and can have both primary and alternate names/codes.
This is typically used to add new species or sound types to the training database.

Args:

- `db_path` *(str, optional)* — Path to the training database. Defaults to cfg.train.train_db.
- `category` *(str)* — Name of the category this class belongs to. Defaults to "default".
- `name` *(str)* — Primary name of the class (e.g., "Common Yellowthroat").
- `code` *(str)* — Primary code for the class (e.g., "COYE").
- `alt_name` *(str, optional)* — Alternate name for the class (e.g., scientific name).
- `alt_code` *(str, optional)* — Alternate code for the class (e.g., scientific code).



### add_src
**Function**  
```python
add_src_cmd(db_path, name)
```
Add a source record to the training database.

Sources track where audio recordings originated from, such as "Xeno-Canto",
"Macaulay Library", "iNaturalist", or custom field recordings. This helps
maintain provenance and can be useful for data quality analysis.

Args:

- `db_path` *(str, optional)* — Path to the training database. Defaults to cfg.train.train_db.
- `name` *(str)* — Name of the source to add (e.g., "Xeno-Canto", "Macaulay Library").



### add_stype
**Function**  
```python
add_stype_cmd(db_path, name)
```
Add a sound type record to the training database.

Sound types describe the nature of the audio content, such as "Song", "Call",
"Alarm", "Drumming", etc. This helps categorize different types of vocalizations
or sounds produced by the same species.

Args:

- `db_path` *(str, optional)* — Path to the training database. Defaults to cfg.train.train_db.
- `name` *(str)* — Name of the sound type to add (e.g., "Song", "Call", "Alarm").



### analyze
**Function**  
```python
analyze_cmd(cfg_path: str, input_path: str, output_path: str, rtype: str, min_score: Optional[float] = None, num_threads: Optional[int] = None, overlap: Optional[float] = None, segment_len: Optional[float] = None)
```
Run inference on audio recordings to detect and classify sounds.

This command processes audio files or directories and generates predictions
using a trained model or ensemble. The output can be saved as Audacity labels,
CSV files, or both.

Args:

- `cfg_path` *(str)* — Path to YAML configuration file defining model and inference settings.
- `input_path` *(str)* — Path to input audio file or directory containing audio files.
- `output_path` *(str)* — Path to output directory where results will be saved.
- `rtype` *(str)* — Output format type. Options are "audacity", "csv", or "both".
- `min_score` *(float, optional)* — Confidence threshold. Predictions below this value are excluded.
- `num_threads` *(int, optional)* — Number of threads to use for processing. Default is 3.
- `overlap` *(float, optional)* — Spectrogram overlap in seconds for sliding window analysis.
- `segment_len` *(float, optional)* — Fixed segment length in seconds. If specified, labels are fixed-length; otherwise they are variable-length.



### audioset
**Function**  
```python
audioset_cmd(class_name: str, curated_csv_path: str, output_dir: str, max_downloads: int, sampling_rate: float, num_to_skip: int, do_report: bool)
```
Download audio recordings from Google AudioSet.

This command downloads audio clips from Google AudioSet, a large-scale dataset of audio events.
You can either download a curated set of recordings or search for a specific audio class.
When using --rpt flag with a class name, it generates a report on associated secondary classes
instead of downloading recordings.

Most AudioSet clips contain multiple classes (e.g., "train", "wind", "speech"). The report
shows which other classes commonly co-occur with the specified class.

Args:

- `class_name` *(str)* — Name of the audio class to download (e.g., "train", "speech", "music").
- `curated_csv_path` *(str)* — Path to CSV file containing a curated list of clips to download.
- `output_dir` *(str)* — Directory where downloaded recordings will be saved.
- `max_downloads` *(int)* — Maximum number of recordings to download. Default is 500.
- `sampling_rate` *(float)* — Output sampling rate in Hz. Default is 32000.
- `num_to_skip` *(int)* — Number of initial recordings to skip. Default is 0.
- `do_report` *(bool)* — If True, generate a report on associated secondary classes instead of downloading.



### ckpt_avg
**Function**  
```python
ckpt_avg_cmd(input_path: str, output_path: str)
```
Average the weights of multiple model checkpoints to create an ensemble checkpoint.

This command loads multiple checkpoint files from a directory and creates a new checkpoint
with averaged weights.

Args:

- `input_path` *(str)* — Directory containing checkpoint files (*.ckpt) to average.
- `output_path` *(str, optional)* — Path for the output averaged checkpoint. Defaults to "average.ckpt" in the input directory.



### ckpt_freeze
**Function**  
```python
ckpt_freeze_cmd(input_path: str)
```
Freeze the backbone weights of a checkpoint to reduce file size and improve inference speed.

This command loads a PyTorch checkpoint and freezes the backbone weights, which removes
training-specific information like gradients and optimizer states. This significantly
reduces the checkpoint file size and can improve inference performance.

The original checkpoint is preserved with a ".original" extension, and the frozen
version replaces the original file. Frozen checkpoints are optimized for deployment
and inference rather than continued training.

Args:

- `input_path` *(str)* — Path to the checkpoint file to freeze.



### ckpt_onnx
**Function**  
```python
ckpt_onnx_cmd(cfg_path: str, input_path: str)
```
Convert a PyTorch checkpoint to ONNX format for deployment with OpenVINO.

This command converts a trained PyTorch model checkpoint to ONNX (Open Neural Network
Exchange) format, which enables deployment using Intel's OpenVINO toolkit. ONNX format
allows for optimized inference on CPU.

The conversion process creates a new ONNX file with the same base name as the input
checkpoint.

Args:

- `cfg_path` *(str, optional)* — Path to YAML file defining configuration overrides.
- `input_path` *(str)* — Path to the PyTorch checkpoint file to convert.



### copy_samples
**Function**  
```python
copy_samples_cmd(dest: pathlib.Path, pattern: str, list_only: bool, overwrite: bool)
```
Copy packaged BriteKit sample YAML/CSV files to a destination directory.

This command copies files from the built-in `britekit.samples` package
(kept alongside the library code) into a folder you specify. You can filter
by subpaths using a simple glob pattern and optionally do a dry-run list.

Args:

- `dest` *(Path)* — Directory to copy sample files into. Subdirectories are created as needed.
- `pattern` *(str)* — Glob-like filter relative to the `samples/` root inside the package.
- `Examples` — `"full/*.yaml"`, `"data/*.csv"`, or `"*"` for everything.
- `list_only` *(bool)* — If True, only list files that match the pattern and exit.
- `overwrite` *(bool)* — If True, overwrite files at the destination when they already exist.
- `Examples` — britekit copy-samples --dest ./examples britekit copy-samples --dest ./examples --pattern 'data/*.csv' britekit copy-samples --dest ./examples --list

Examples:
    britekit copy-samples --dest ./examples
    britekit copy-samples --dest ./examples --pattern 'data/*.csv'
    britekit copy-samples --dest ./examples --list

### del_cat
**Function**  
```python
del_cat_cmd(db_path, name)
```
Delete a category and all its associated data from the training database.

This command performs a cascading delete that removes the category, all its classes,
all recordings belonging to those classes, and all spectrograms from those recordings.
This is a destructive operation that cannot be undone.

Args:

- `db_path` *(str, optional)* — Path to the training database. Defaults to cfg.train.train_db.
- `name` *(str)* — Name of the category to delete (e.g., "Birds", "Mammals").



### del_class
**Function**  
```python
del_class_cmd(db_path, class_name)
```
Delete a class and all its associated data from the training database.

This command removes the class, all recordings belonging to that class, and all
spectrograms from those recordings. This is a destructive operation that cannot
be undone and will affect any training data associated with this class.

Args:

- `db_path` *(str, optional)* — Path to the training database. Defaults to cfg.train.train_db.
- `class_name` *(str)* — Name of the class to delete (e.g., "Common Yellowthroat").



### del_rec
**Function**  
```python
del_rec_cmd(db_path, file_name)
```
Delete a recording and all its spectrograms from the training database.

This command removes a specific audio recording and all spectrograms that were
extracted from it.

Args:

- `db_path` *(str, optional)* — Path to the training database. Defaults to cfg.train.train_db.
- `file_name` *(str)* — Name of the recording file to delete (e.g., "XC123456.mp3").



### del_spec
**Function**  
```python
del_spec_cmd(db_path, class_name, dir_path)
```
Delete spectrograms that correspond to images in a given directory.

This command parses image filenames to identify and delete corresponding spectrograms
from the database. Images are typically generated by the plot-db or search commands,
and their filenames contain the recording name and time offset.

This is useful for removal of spectrograms based on visual inspection of plots,
allowing you to remove low-quality or incorrectly labeled spectrograms.

Args:

- `db_path` *(str, optional)* — Path to the training database. Defaults to cfg.train.train_db.
- `class_name` *(str)* — Name of the class whose spectrograms should be considered for deletion.
- `dir_path` *(str)* — Path to directory containing spectrogram image files (typically .jpeg files).



### del_src
**Function**  
```python
del_src_cmd(db_path, name)
```
Delete a recording source and all its associated data from the training database.

This command performs a cascading delete that removes the source, all recordings
from that source, and all spectrograms from those recordings. This is useful for
removing entire datasets from a specific source (e.g., removing all Xeno-Canto data).

Args:

- `db_path` *(str, optional)* — Path to the training database. Defaults to cfg.train.train_db.
- `name` *(str)* — Name of the source to delete (e.g., "Xeno-Canto", "Macaulay Library").



### del_stype
**Function**  
```python
del_stype_cmd(db_path, name)
```
Delete a sound type from the training database.

This command removes a sound type definition but preserves the spectrograms that were
labeled with this sound type. The spectrograms will have their soundtype_id field set
to null, effectively removing the sound type classification while keeping the audio data.

Args:

- `db_path` *(str, optional)* — Path to the training database. Defaults to cfg.train.train_db.
- `name` *(str)* — Name of the sound type to delete (e.g., "Song", "Call", "Alarm").



### embed
**Function**  
```python
embed_cmd(cfg_path: str, db_path: str, class_name: str, spec_group: str)
```
Generate embeddings for spectrograms and insert them into the database.

This command uses a trained model to generate embeddings (feature vectors) for spectrograms
in the training database. These embeddings can be used for similarity search and other
downstream tasks. The embeddings are compressed and stored in the database.

Args:

- `cfg_path` *(str, optional)* — Path to YAML file defining configuration overrides.
- `db_path` *(str, optional)* — Path to the training database. Defaults to cfg.train.train_db.
- `class_name` *(str, optional)* — Name of a specific class to process. If omitted, processes all classes.
- `spec_group` *(str)* — Spectrogram group name to process. Defaults to 'default'.



### extract_all
**Function**  
```python
extract_all_cmd(cfg_path: str, db_path: str, cat_name: str, class_code: str, class_name: str, dir_path: str, overlap: float, src_name: str, spec_group: str)
```
Extract all spectrograms from audio recordings and insert them into the training database.

This command processes all audio files in a directory and extracts spectrograms using
sliding windows with optional overlap. The spectrograms are then inserted into the
training database for use in model training. If the specified class doesn't exist,
it will be automatically created.

Args:

- `cfg_path` *(str, optional)* — Path to YAML file defining configuration overrides.
- `db_path` *(str, optional)* — Path to the training database. Defaults to cfg.train.train_db.
- `cat_name` *(str, optional)* — Category name for new class creation (e.g., "bird"). Defaults to "default".
- `class_code` *(str, optional)* — Class code for new class creation (e.g., "COYE").
- `class_name` *(str)* — Name of the class for the recordings (e.g., "Common Yellowthroat").
- `dir_path` *(str)* — Path to directory containing audio recordings to process.
- `overlap` *(float, optional)* — Spectrogram overlap in seconds. Defaults to config value.
- `src_name` *(str, optional)* — Source name for the recordings (e.g., "Xeno-Canto"). Defaults to "default".
- `spec_group` *(str, optional)* — Spectrogram group name for organizing extractions. Defaults to "default".



### extract_by_image
**Function**  
```python
extract_by_image_cmd(cfg_path: str, db_path: str, cat_name: str, class_code: str, class_name: str, rec_dir: str, spec_dir: str, dest_dir: str, src_name: str, spec_group: str)
```
Extract spectrograms that correspond to existing spectrogram images.

This command parses spectrogram image filenames to identify the corresponding audio
segments and extracts those specific spectrograms from the original recordings.
This is useful when you have pre-selected spectrograms (e.g., from manual review
or search results) and want to extract only those specific segments.

The images contain metadata in their filenames (recording name and time offset)
that allows the command to locate and extract the corresponding audio segments.

Args:

- `cfg_path` *(str, optional)* — Path to YAML file defining configuration overrides.
- `db_path` *(str, optional)* — Path to the training database. Defaults to cfg.train.train_db.
- `cat_name` *(str, optional)* — Category name for new class creation (e.g., "bird"). Defaults to "default".
- `class_code` *(str, optional)* — Class code for new class creation (e.g., "COYE").
- `class_name` *(str)* — Name of the class for the recordings (e.g., "Common Yellowthroat").
- `rec_dir` *(str)* — Path to directory containing the original audio recordings.
- `spec_dir` *(str)* — Path to directory containing spectrogram image files.
- `dest_dir` *(str, optional)* — If specified, copy used recordings to this directory.
- `src_name` *(str, optional)* — Source name for the recordings (e.g., "Xeno-Canto"). Defaults to "default".
- `spec_group` *(str, optional)* — Spectrogram group name for organizing extractions. Defaults to "default".



### find_dup
**Function**  
```python
find_dup_cmd(cfg_path: str, db_path: str, class_name: str, delete: bool, spec_group: str)
```
Find and optionally delete duplicate recordings in the training database.

This command scans the database for recordings of the same class that appear to be duplicates.
It uses a two-stage detection approach:
1. Compare recording durations (within 0.1 seconds tolerance)
2. Compare spectrogram embeddings of the first few spectrograms (within 0.02 cosine distance)

Duplicates are identified by comparing the first 3 spectrogram embeddings from each recording
using cosine distance.

Args:

- `cfg_path` *(str, optional)* — Path to YAML file defining configuration overrides.
- `db_path` *(str, optional)* — Path to the training database. Defaults to cfg.train.train_db.
- `class_name` *(str)* — Name of the class to scan for duplicates (e.g., "Common Yellowthroat").
- `delete` *(bool)* — If True, remove duplicate recordings from the database. If False, only report them.
- `spec_group` *(str)* — Spectrogram group name to use for embedding comparison. Defaults to "default".



### inat
**Function**  
```python
inat_cmd(output_dir: str, max_downloads: int, name: str, no_prefix: bool)
```
Download audio recordings from iNaturalist observations.

This command searches iNaturalist for observations of a specified species that contain
audio recordings. It downloads the audio files and creates a CSV file mapping the
downloaded files to their iNaturalist observation URLs for reference.

Only observations with "research grade" quality are downloaded (excluding "needs_id").
The command respects the maximum download limit and can optionally add filename prefixes.

Args:

- `output_dir` *(str)* — Directory where downloaded recordings will be saved.
- `max_downloads` *(int)* — Maximum number of recordings to download. Default is 500.
- `name` *(str)* — Species name to search for (e.g., "Common Yellowthroat", "Geothlypis trichas").
- `no_prefix` *(bool)* — If True, skip adding "N" prefix to filenames. Default adds prefix.



### pickle
**Function**  
```python
pickle_cmd(cfg_path, classes_path, db_path, output_path, max_per_class, spec_group)
```
Convert database spectrograms to a pickle file for use in training.

This command extracts spectrograms from the training database and saves them in a pickle file
that can be efficiently loaded during model training. It can process all classes in the database
or specific classes specified by a CSV file.

Args:

- `cfg_path` *(str, optional)* — Path to YAML file defining configuration overrides.
- `classes_path` *(str, optional)* — Path to CSV file containing class names to include. If omitted, includes all classes in the database.
- `db_path` *(str, optional)* — Path to the training database. Defaults to cfg.train.train_db.
- `output_path` *(str, optional)* — Output pickle file path. Defaults to "data/training.pkl".
- `max_per_class` *(int, optional)* — Maximum number of spectrograms to include per class.
- `spec_group` *(str)* — Spectrogram group name to extract from. Defaults to 'default'.



### plot_db
**Function**  
```python
plot_db_cmd(cfg_path: str, class_name: str, db_path: Optional[str], dims: bool, max_count: Optional[float], output_path: str, prefix: Optional[str], power: Optional[float], spec_group: Optional[str])
```
Plot spectrograms from a training database for a specific class.

This command extracts spectrograms from the training database for a given class and
saves them as JPEG images. It can filter recordings by filename prefix and limit the
number of spectrograms plotted.

Args:

- `cfg_path` *(str, optional)* — Path to YAML file defining configuration overrides.
- `class_name` *(str)* — Name of the class to plot spectrograms for (e.g., "Common Yellowthroat").
- `db_path` *(str, optional)* — Path to the training database. Defaults to cfg.train.train_db.
- `dims` *(bool)* — If True, show time and frequency dimensions on the spectrogram plots.
- `max_count` *(int, optional)* — Maximum number of spectrograms to plot. If omitted, plots all available.
- `output_path` *(str)* — Directory where spectrogram images will be saved.
- `prefix` *(str, optional)* — Only include recordings that start with this filename prefix.
- `power` *(float, optional)* — Raise spectrograms to this power for visualization. Lower values show more detail.
- `spec_group` *(str, optional)* — Spectrogram group name to plot from. Defaults to "default".



### plot_dir
**Function**  
```python
plot_dir_cmd(cfg_path: str, dims: bool, input_path: str, output_path: str, all: bool, overlap: float, power: float = 1.0)
```
Plot spectrograms for all audio recordings in a directory.

This command processes all audio files in a directory and generates spectrogram images.
It can either plot each recording as a single spectrogram or break recordings into
overlapping segments.

Args:

- `cfg_path` *(str, optional)* — Path to YAML file defining configuration overrides.
- `dims` *(bool)* — If True, show time and frequency dimensions on the spectrogram plots.
- `input_path` *(str)* — Directory containing audio recordings to process.
- `output_path` *(str)* — Directory where spectrogram images will be saved.
- `all` *(bool)* — If True, plot each recording as one spectrogram. If False, break into segments.
- `overlap` *(float)* — Spectrogram overlap in seconds when breaking recordings into segments. Default is 0.
- `power` *(float)* — Raise spectrograms to this power for visualization. Lower values show more detail. Default is 1.0.



### reextract
**Function**  
```python
reextract_cmd(cfg_path: Optional[str] = None, db_path: Optional[str] = None, class_name: Optional[str] = None, class_csv_path: Optional[str] = None, check: bool = False, spec_group: str = 'default')
```
Re-generate spectrograms from audio recordings and update the training database.

This command extracts spectrograms from audio recordings and imports them into the training database.
It can process all classes in the database or specific classes specified by name or CSV file.
If the specified spectrogram group already exists, it will be deleted and recreated.

In check mode, it only verifies that all required audio files are accessible without
updating the database.

Args:

- `cfg_path` *(str, optional)* — Path to YAML file defining configuration overrides.
- `db_path` *(str, optional)* — Path to the training database. Defaults to cfg.train.training_db.
- `class_name` *(str, optional)* — Name of a specific class to reextract. If omitted, processes all classes.
- `class_csv_path` *(str, optional)* — Path to CSV file listing classes to reextract. Alternative to class_name.
- `check` *(bool)* — If True, only check that all recording paths are accessible without updating database.
- `spec_group` *(str)* — Spectrogram group name for storing the extracted spectrograms. Defaults to 'default'.



### rpt_ann
**Function**  
```python
rpt_ann_cmd(annotations_path: str, output_path: str)
```
Summarize per-segment annotations from a test dataset.

This command reads annotation data from a CSV file and generates summary reports
showing the total duration of each class across all recordings and per-recording
breakdowns.

Args:

- `annotations_path` *(str)* — Path to CSV file containing per-segment annotations.
- `output_path` *(str)* — Directory where summary reports will be saved.



### rpt_cal
**Function**  
```python
rpt_cal_cmd(cfg_path: str, annotations_path: str, label_dir: str, output_path: str, class_csv_path: str, recordings_path: Optional[str], cutoff: float, coef: Optional[float] = None, inter: Optional[float] = None)
```
Calibrate model predictions using per-segment test results.

This command generates calibration plots and analysis to assess how well
model prediction scores align with actual probabilities. It compares
predicted scores against ground truth annotations to determine if the
model is overconfident or underconfident in its predictions.

The calibration process helps improve model reliability by adjusting
prediction scores to better reflect true probabilities.

Args:

- `cfg_path` *(str, optional)* — Path to YAML file defining configuration overrides.
- `annotations_path` *(str)* — Path to CSV file containing ground truth annotations.
- `label_dir` *(str)* — Directory containing model prediction labels (Audacity format).
- `output_path` *(str)* — Directory where calibration reports will be saved.
- `class_csv_path` *(str)* — Path to CSV file listing classes included in training.
- `recordings_path` *(str, optional)* — Directory containing audio recordings. Defaults to annotations directory.
- `cutoff` *(float)* — Ignore predictions below this threshold during calibration. Default is 0.4.
- `coef` *(float, optional)* — Use this coefficient for the calibration plot.
- `inter` *(float, optional)* — Use this intercept for the calibration plot.



### rpt_db
**Function**  
```python
rpt_db_cmd(cfg_path, db_path, output_path)
```
Generate a comprehensive summary report of the training database.

This command analyzes the training database and generates detailed reports
about class distributions, spectrogram groups, and data organization.
The reports help understand the composition and quality of training data
and can be used for data management and quality control.

Args:

- `cfg_path` *(str, optional)* — Path to YAML file defining configuration overrides.
- `db_path` *(str, optional)* — Path to the training database. Defaults to cfg.train.train_db.
- `output_path` *(str)* — Directory where database reports will be saved.



### rpt_labels
**Function**  
```python
rpt_labels_cmd(label_dir: str, output_path: str, min_score: Optional[float])
```
Summarize the output of an inference run.

This command processes inference results (from CSV files or Audacity labels)
and generates summary reports showing the total duration of detections
per class and per recording. It filters results by confidence threshold
and removes overlapping detections to provide clean statistics.

The reports help understand model performance and detection patterns
across different recordings and classes.

Args:

- `label_dir` *(str)* — Directory containing inference output (CSV or Audacity labels).
- `output_path` *(str)* — Directory where summary reports will be saved.
- `min_score` *(float, optional)* — Ignore detections below this confidence threshold.



### rpt_test
**Function**  
```python
rpt_test_cmd(cfg_path: str, granularity: str, annotations_path: str, label_dir: str, output_path: str, class_csv_path: str, recordings_path: Optional[str], min_score: Optional[float], precision: float)
```
Generate comprehensive test metrics and reports comparing model predictions to ground truth.

This command evaluates model performance by comparing inference results against
ground truth annotations. It supports three granularity levels:
- "recording": Evaluate at the recording level (presence/absence)
- "minute": Evaluate at the minute level (presence/absence per minute)
- "segment": Evaluate at the segment level (detailed temporal alignment)

The command generates detailed performance metrics including precision, recall,
F1 scores, and various visualization plots to help understand model behavior.

Args:

- `cfg_path` *(str, optional)* — Path to YAML file defining configuration overrides.
- `granularity` *(str)* — Evaluation granularity ("recording", "minute", or "segment"). Default is "segment".
- `annotations_path` *(str)* — Path to CSV file containing ground truth annotations.
- `label_dir` *(str)* — Directory containing model prediction labels (Audacity format).
- `output_path` *(str)* — Directory where test reports will be saved.
- `class_csv_path` *(str)* — Path to CSV file listing classes included in training.
- `recordings_path` *(str, optional)* — Directory containing audio recordings. Defaults to annotations directory.
- `min_score` *(float, optional)* — Provide detailed reports for this confidence threshold.
- `precision` *(float)* — For recording granularity, report true positive seconds at this precision. Default is 0.95.



### search
**Function**  
```python
search_cmd(cfg_path: str, db_path: str, class_name: str, max_dist: float, exp: float, num_to_plot: int, output_path: str, input_path: str, offset: float, exclude_db: str, class_name2: str, spec_group: str)
```
Search a database for spectrograms similar to a specified one.

This command extracts a spectrogram from a given audio file at a specified offset,
then searches through a database of spectrograms to find the most similar ones
based on embedding similarity. Results are plotted and saved to the output directory.

Args:

- `cfg_path` *(str)* — Path to YAML configuration file defining model settings.
- `db_path` *(str)* — Path to the training database containing spectrograms to search.
- `class_name` *(str)* — Name of the class/species to search within the database.
- `max_dist` *(float)* — Maximum distance threshold. Results with distance greater than this are excluded.
- `exp` *(float)* — Exponent to raise spectrograms to for visualization (shows background sounds).
- `num_to_plot` *(int)* — Maximum number of similar spectrograms to plot and save.
- `output_path` *(str)* — Directory where search results and plots will be saved.
- `input_path` *(str)* — Path to the audio file containing the target spectrogram.
- `offset` *(float)* — Time offset in seconds where the target spectrogram is extracted.
- `exclude_db` *(str, optional)* — Path to an exclusion database. Spectrograms in this database are excluded from results.
- `class_name2` *(str, optional)* — Class name in the exclusion database. Defaults to the search class name.
- `spec_group` *(str)* — Spectrogram group name in the database. Defaults to 'default'.



### train
**Function**  
```python
train_cmd(cfg_path: str)
```
Train a bioacoustic recognition model using the specified configuration.

This command initiates the complete training pipeline for a bioacoustic model.
It loads training data from the database, configures the model architecture,
and runs the training process with the specified hyperparameters. The training
includes validation, checkpointing, and progress monitoring.

Training progress is displayed in real-time, and model checkpoints are saved
automatically. The final trained model can be used for inference and evaluation.

Args:

- `cfg_path` *(str, optional)* — Path to YAML file defining configuration overrides. If not specified, uses default configuration.



### tune
**Function**  
```python
tune_cmd(cfg_path: str, param_path: str, annotations_path: str, class_csv_path: str, metric: str, recordings_path: str, train_log_path: str, num_trials: int, num_runs: int)
```
Find and print the best hyperparameter settings based on exhaustive or random search.

This command performs hyperparameter optimization by training models with different
parameter combinations and evaluating them using the specified metric. It can perform
either exhaustive search (testing all combinations) or random search (testing a
specified number of random combinations).

The param_path specifies a YAML file that contains a sequence of parameters such as:

- name: prob_simple_merge
  type: float
  bounds:
  - 0.36
  - 0.42
  step: 0.02

The name defines a hyperparameter to tune, with given type, bounds and step size.

Args:

- `cfg_path` *(str)* — Path to YAML file defining configuration overrides.
- `param_path` *(str)* — Path to YAML file defining hyperparameters to tune and their search space.
- `annotations_path` *(str)* — Path to CSV file containing ground truth annotations.
- `class_csv_path` *(str)* — Path to CSV file listing classes included in training.
- `metric` *(str)* — Metric used to compare runs. Options include various MAP and ROC metrics.
- `recordings_path` *(str, optional)* — Directory containing audio recordings. Defaults to annotations directory.
- `train_log_path` *(str, optional)* — Training log directory. Defaults to "logs/fold-0".
- `num_trials` *(int)* — Number of random trials to run. If 0, performs exhaustive search.
- `num_runs` *(int)* — Number of runs to average for each parameter combination. Default is 1.



### wav2mp3
**Function**  
```python
wav2mp3_cmd(dir: str, sampling_rate: int)
```
Convert uncompressed audio files to MP3 format and replace the originals.

This command processes all uncompressed audio files in a directory and converts
them to MP3 format using FFmpeg. Supported input formats include FLAC, WAV, WMA,
AIFF, and other uncompressed audio formats. After successful conversion, the
original files are deleted to save disk space.

The conversion uses a 192k bitrate for good quality while maintaining reasonable
file sizes. This is useful for standardizing audio formats and reducing storage
requirements for large audio datasets.

Args:

- `dir` *(str)* — Path to directory containing audio files to convert.
- `sampling_rate` *(int)* — Output sampling rate in Hz. Default is 32000 Hz.



### xeno
**Function**  
```python
xeno_cmd(key: str, output_dir: str, max_downloads: int, name: str, ignore_licence: bool, scientific_name: bool, seen_only: bool)
```
Download bird song recordings from Xeno-Canto database.

This command uses the Xeno-Canto API v3 to search for and download audio recordings
of bird songs. The API requires authentication via an API key. Recordings are
downloaded as MP3 files and saved to the specified output directory.

To get an API key, register as a Xeno-Canto user and check your account page.
Then specify the key in the --key argument, or set the environment variable XCKEY=<key>.

Args:

- `key` *(str)* — Xeno-Canto API key for authentication. Can also be set via XCKEY environment variable.
- `output_dir` *(str)* — Directory where downloaded recordings will be saved.
- `max_downloads` *(int)* — Maximum number of recordings to download. Default is 500.
- `name` *(str)* — Species name to search for (common name or scientific name).
- `ignore_licence` *(bool)* — If True, ignore license restrictions. By default, excludes BY-NC-ND licensed recordings.
- `scientific_name` *(bool)* — If True, treat the name as a scientific name rather than common name.
- `seen_only` *(bool)* — If True, only download recordings where the animal was seen (animal-seen=yes).
