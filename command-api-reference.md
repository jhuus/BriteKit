## Command API Reference
### add_cat
**Function**  
```python
add_cat(db_path: Optional[str] = None, name: str = '') -> None
```
Add a category (class group) record to the training database.

Categories are used to group related classes together in the database.
For example, you might have categories like "Birds", "Mammals", or "Insects"
that contain multiple related species classes.

Args:
    db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    name (str): Name of the category to add (e.g., "Birds", "Mammals").

### add_class
**Function**  
```python
add_class(db_path: Optional[str] = None, category: str = 'default', name: Optional[str] = None, code: Optional[str] = None, alt_name: Optional[str] = None, alt_code: Optional[str] = None) -> None
```
Add a class record to the training database.

Classes represent the target species or sound categories for training and inference.
Each class must belong to a category and can have both primary and alternate names/codes.
This is typically used to add new species or sound types to the training database.

Args:
    db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    category (str): Name of the category this class belongs to. Defaults to "default".
    name (str): Primary name of the class (e.g., "Common Yellowthroat").
    code (str): Primary code for the class (e.g., "COYE").
    alt_name (str, optional): Alternate name for the class (e.g., scientific name).
    alt_code (str, optional): Alternate code for the class (e.g., scientific code).

### add_src
**Function**  
```python
add_src(db_path: Optional[str] = None, name: str = '') -> None
```
Add a source record to the training database.

Sources track where audio recordings originated from, such as "Xeno-Canto",
"Macaulay Library", "iNaturalist", or custom field recordings. This helps
maintain provenance and can be useful for data quality analysis.

Args:
    db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    name (str): Name of the source to add (e.g., "Xeno-Canto", "Macaulay Library").

### add_stype
**Function**  
```python
add_stype(db_path: Optional[str] = None, name: str = '') -> None
```
Add a sound type record to the training database.

Sound types describe the nature of the audio content, such as "Song", "Call",
"Alarm", "Drumming", etc. This helps categorize different types of vocalizations
or sounds produced by the same species.

Args:
    db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    name (str): Name of the sound type to add (e.g., "Song", "Call", "Alarm").

### analyze
**Function**  
```python
analyze(cfg_path: Optional[str] = None, input_path: str = '', output_path: str = '', rtype: str = 'both', min_score: Optional[float] = None, num_threads: Optional[int] = None, overlap: Optional[float] = None, segment_len: Optional[float] = None)
```
Run inference on audio recordings to detect and classify sounds.

This command processes audio files or directories and generates predictions
using a trained model or ensemble. The output can be saved as Audacity labels,
CSV files, or both.

Args:
    cfg_path (str): Path to YAML configuration file defining model and inference settings.
    input_path (str): Path to input audio file or directory containing audio files.
    output_path (str): Path to output directory where results will be saved.
    rtype (str): Output format type. Options are "audacity", "csv", or "both".
    min_score (float, optional): Confidence threshold. Predictions below this value are excluded.
    num_threads (int, optional): Number of threads to use for processing. Default is 3.
    overlap (float, optional): Spectrogram overlap in seconds for sliding window analysis.
    segment_len (float, optional): Fixed segment length in seconds. If specified, labels are
                                 fixed-length; otherwise they are variable-length.

### audioset
**Function**  
```python
audioset(class_name: Optional[str] = None, curated_csv_path: Optional[str] = None, output_dir: str = '', max_downloads: int = 500, sampling_rate: int = 32000, num_to_skip: int = 0, do_report: bool = False, root_dir: str = '.') -> None
```
Download audio recordings from Google AudioSet.

This command downloads audio clips from Google AudioSet, a large-scale dataset of audio events.
You can either download a curated set of recordings or search for a specific audio class.
When using --rpt flag with a class name, it generates a report on associated secondary classes
instead of downloading recordings.

Most AudioSet clips contain multiple classes (e.g., "train", "wind", "speech"). The report
shows which other classes commonly co-occur with the specified class.

Args:
    class_name (str): Name of the audio class to download (e.g., "train", "speech", "music").
    curated_csv_path (str): Path to CSV file containing a curated list of clips to download.
    output_dir (str): Directory where downloaded recordings will be saved.
    max_downloads (int): Maximum number of recordings to download. Default is 500.
    sampling_rate (float): Output sampling rate in Hz. Default is 32000.
    num_to_skip (int): Number of initial recordings to skip. Default is 0.
    do_report (bool): If True, generate a report on associated secondary classes instead of downloading.
    root_dir (str): Directory that contains the data directory. Default is working directory.

### calibrate
**Function**  
```python
calibrate(cfg_path: Optional[str] = None, annotations_path: str = '', label_dir: str = '', output_path: str = '', recordings_path: Optional[str] = None, cutoff: float = 0.4, coef: Optional[float] = None, inter: Optional[float] = None)
```
Calibrate model predictions using per-segment test results.

This command generates calibration plots and analysis to assess how well
model prediction scores align with actual probabilities. It compares
predicted scores against ground truth annotations to determine if the
model is overconfident or underconfident in its predictions.

The calibration process helps improve model reliability by adjusting
prediction scores to better reflect true probabilities.

Args:
    cfg_path (str, optional): Path to YAML file defining configuration overrides.
    annotations_path (str): Path to CSV file containing ground truth annotations.
    label_dir (str): Directory containing model prediction labels (Audacity format).
    output_path (str): Directory where calibration reports will be saved.
    recordings_path (str, optional): Directory containing audio recordings. Defaults to annotations directory.
    cutoff (float): Ignore predictions below this threshold during calibration. Default is 0.4.
    coef (float, optional): Use this coefficient for the calibration plot.
    inter (float, optional): Use this intercept for the calibration plot.

### ckpt_avg
**Function**  
```python
ckpt_avg(input_path: str = '', output_path: Optional[str] = None)
```
Average the weights of multiple model checkpoints to create an ensemble checkpoint.

This command loads multiple checkpoint files from a directory and creates a new checkpoint
with averaged weights.

Args:
    input_path (str): Directory containing checkpoint files (*.ckpt) to average.
    output_path (str, optional): Path for the output averaged checkpoint.
                               Defaults to "average.ckpt" in the input directory.

### ckpt_freeze
**Function**  
```python
ckpt_freeze(input_path: str = '')
```
Freeze the backbone weights of a checkpoint to reduce file size and improve inference speed.

This command loads a PyTorch checkpoint and freezes the backbone weights, which removes
training-specific information like gradients and optimizer states. This significantly
reduces the checkpoint file size and can improve inference performance.

The original checkpoint is preserved with a ".original" extension, and the frozen
version replaces the original file. Frozen checkpoints are optimized for deployment
and inference rather than continued training.

Args:
    input_path (str): Path to the checkpoint file to freeze.

### ckpt_onnx
**Function**  
```python
ckpt_onnx(cfg_path: Optional[str] = None, input_path: str = '')
```
Convert a PyTorch checkpoint to ONNX format for deployment with OpenVINO.

This command converts a trained PyTorch model checkpoint to ONNX (Open Neural Network
Exchange) format, which enables deployment using Intel's OpenVINO toolkit. ONNX format
allows for optimized inference on CPU.

The conversion process creates a new ONNX file with the same base name as the input
checkpoint.

Args:
    cfg_path (str, optional): Path to YAML file defining configuration overrides.
    input_path (str): Path to the PyTorch checkpoint file to convert.

### del_cat
**Function**  
```python
del_cat(db_path: Optional[str] = None, name: Optional[str] = None) -> None
```
Delete a category and all its associated data from the training database.

This command performs a cascading delete that removes the category, all its classes,
all recordings belonging to those classes, and all spectrograms from those recordings.
This is a destructive operation that cannot be undone.

Args:
    db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    name (str): Name of the category to delete (e.g., "Birds", "Mammals").

### del_class
**Function**  
```python
del_class(db_path: Optional[str] = None, name: Optional[str] = None) -> None
```
Delete a class and all its associated data from the training database.

This command removes the class, all recordings belonging to that class, and all
spectrograms from those recordings. This is a destructive operation that cannot
be undone and will affect any training data associated with this class.

Args:
    db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    name (str): Name of the class to delete (e.g., "Common Yellowthroat").

### del_rec
**Function**  
```python
del_rec(db_path: Optional[str] = None, file_name: Optional[str] = None) -> None
```
Delete a recording and all its spectrograms from the training database.

This command removes a specific audio recording and all spectrograms that were
extracted from it.

Args:
    db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    file_name (str): Name of the recording file to delete (e.g., "XC123456.mp3").

### del_seg
**Function**  
```python
del_seg(db_path: Optional[str] = None, class_name: Optional[str] = None, dir_path: Optional[str] = None) -> None
```
Delete segments that correspond to images in a given directory.

This command parses image filenames to identify and delete corresponding segments
from the database. Images are typically generated by the plot-db or search commands,
and their filenames contain the recording name and time offset.

This is useful for removal of segments based on visual inspection of plots,
allowing you to remove low-quality or incorrectly labeled segments.

Args:
    db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    class_name (str): Name of the class whose segments should be considered for deletion.
    dir_path (str): Path to directory containing spectrogram image files.

### del_sgroup
**Function**  
```python
del_sgroup(db_path: Optional[str] = None, name: Optional[str] = None) -> None
```
Delete a spectrogram group and all its spectrogram values from the training database.

Spectrogram groups organize spectrograms by processing parameters or extraction method.
This command removes the entire group and all spectrograms within it.

Args:
    db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    name (str): Name of the spectrogram group to delete (e.g., "default", "augmented").

### del_src
**Function**  
```python
del_src(db_path: Optional[str] = None, name: Optional[str] = None) -> None
```
Delete a recording source and all its associated data from the training database.

This command performs a cascading delete that removes the source, all recordings
from that source, and all spectrograms from those recordings. This is useful for
removing entire datasets from a specific source (e.g., removing all Xeno-Canto data).

Args:
    db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    name (str): Name of the source to delete (e.g., "Xeno-Canto", "Macaulay Library").

### del_stype
**Function**  
```python
del_stype(db_path: Optional[str] = None, name: Optional[str] = None) -> None
```
Delete a sound type from the training database.

This command removes a sound type definition but preserves the spectrograms that were
labeled with this sound type. The spectrograms will have their soundtype_id field set
to null, effectively removing the sound type classification while keeping the audio data.

Args:
    db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    name (str): Name of the sound type to delete (e.g., "Song", "Call", "Alarm").

### embed
**Function**  
```python
embed(cfg_path: Optional[str] = None, db_path: Optional[str] = None, class_name: Optional[str] = None, spec_group: str = 'default') -> None
```
Generate embeddings for spectrograms and insert them into the database.

This command uses a trained model to generate embeddings (feature vectors) for spectrograms
in the training database. These embeddings can be used for similarity search and other
downstream tasks. The embeddings are compressed and stored in the database.

Args:
    cfg_path (str, optional): Path to YAML file defining configuration overrides.
    db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    class_name (str, optional): Name of a specific class to process. If omitted, processes all classes.
    spec_group (str): Spectrogram group name to process. Defaults to 'default'.

### extract_all
**Function**  
```python
extract_all(cfg_path: Optional[str] = None, db_path: Optional[str] = None, cat_name: Optional[str] = None, class_code: Optional[str] = None, class_name: str = '', dir_path: str = '', overlap: Optional[float] = None, src_name: Optional[str] = None, spec_group: Optional[str] = None) -> None
```
Extract all spectrograms from audio recordings and insert them into the training database.

This command processes all audio files in a directory and extracts spectrograms using
sliding windows with optional overlap. The spectrograms are then inserted into the
training database for use in model training. If the specified class doesn't exist,
it will be automatically created.

Args:
    cfg_path (str, optional): Path to YAML file defining configuration overrides.
    db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    cat_name (str, optional): Category name for new class creation (e.g., "bird"). Defaults to "default".
    class_code (str, optional): Class code for new class creation (e.g., "COYE").
    class_name (str): Name of the class for the recordings (e.g., "Common Yellowthroat").
    dir_path (str): Path to directory containing audio recordings to process.
    overlap (float, optional): Spectrogram overlap in seconds. Defaults to config value.
    src_name (str, optional): Source name for the recordings (e.g., "Xeno-Canto"). Defaults to "default".
    spec_group (str, optional): Spectrogram group name for organizing extractions. Defaults to "default".

### extract_by_image
**Function**  
```python
extract_by_image(cfg_path: Optional[str] = None, db_path: Optional[str] = None, cat_name: Optional[str] = None, class_code: Optional[str] = None, class_name: str = '', rec_dir: str = '', spec_dir: str = '', dest_dir: Optional[str] = None, src_name: Optional[str] = None, spec_group: Optional[str] = None) -> None
```
Extract spectrograms that correspond to existing spectrogram images.

This command parses spectrogram image filenames to identify the corresponding audio
segments and extracts those specific spectrograms from the original recordings.
This is useful when you have pre-selected spectrograms (e.g., from manual review
or search results) and want to extract only those specific segments.

The images contain metadata in their filenames (recording name and time offset)
that allows the command to locate and extract the corresponding audio segments.

Args:
    cfg_path (str, optional): Path to YAML file defining configuration overrides.
    db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    cat_name (str, optional): Category name for new class creation (e.g., "bird"). Defaults to "default".
    class_code (str, optional): Class code for new class creation (e.g., "COYE").
    class_name (str): Name of the class for the recordings (e.g., "Common Yellowthroat").
    rec_dir (str): Path to directory containing the original audio recordings.
    spec_dir (str): Path to directory containing spectrogram image files.
    dest_dir (str, optional): If specified, copy used recordings to this directory.
    src_name (str, optional): Source name for the recordings (e.g., "Xeno-Canto"). Defaults to "default".
    spec_group (str, optional): Spectrogram group name for organizing extractions. Defaults to "default".

### find_dup
**Function**  
```python
find_dup(cfg_path: Optional[str] = None, db_path: Optional[str] = None, class_name: str = '', delete: bool = False, spec_group: str = 'default') -> None
```
Find and optionally delete duplicate recordings in the training database.

This command scans the database for recordings of the same class that appear to be duplicates.
It uses a two-stage detection approach:
1. Compare recording durations (within 0.1 seconds tolerance)
2. Compare spectrogram embeddings of the first few spectrograms (within 0.02 cosine distance)

Duplicates are identified by comparing the first 3 spectrogram embeddings from each recording
using cosine distance.

Args:
    cfg_path (str, optional): Path to YAML file defining configuration overrides.
    db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    class_name (str): Name of the class to scan for duplicates (e.g., "Common Yellowthroat").
    delete (bool): If True, remove duplicate recordings from the database. If False, only report them.
    spec_group (str): Spectrogram group name to use for embedding comparison. Defaults to "default".

### find_lr
**Function**  
```python
find_lr(cfg_path: str, num_batches: int)
```
Find an optimal learning rate for model training using the learning rate finder.

This command runs a learning rate finder that tests a range of learning rates
on a small number of training batches to determine the optimal learning rate.
It generates a plot showing loss vs. learning rate and suggests the best rate
based on the steepest negative gradient in the loss curve.

The suggested learning rate helps ensure stable and efficient training by
avoiding rates that are too high (causing instability) or too low (slow convergence).

Args:
    cfg_path (str, optional): Path to YAML file defining configuration overrides.
                             If not specified, uses default configuration.
    num_batches (int): Number of training batches to analyze for learning rate finding.
                      Default is 100. Higher values provide more accurate results but take longer.

### inat
**Function**  
```python
inat(name: str = '', output_dir: str = '', max_downloads: int = 500, no_prefix: bool = False) -> None
```
Download audio recordings from iNaturalist observations.

This command searches iNaturalist for observations of a specified species that contain
audio recordings. It downloads the audio files and creates a CSV file mapping the
downloaded files to their iNaturalist observation URLs for reference.

Only observations with "research grade" quality are downloaded (excluding "needs_id").
The command respects the maximum download limit and can optionally add filename prefixes.

Args:
    output_dir (str): Directory where downloaded recordings will be saved.
    max_downloads (int): Maximum number of recordings to download. Default is 500.
    name (str): Species name to search for (e.g., "Common Yellowthroat", "Geothlypis trichas").
    no_prefix (bool): If True, skip adding "N" prefix to filenames. Default adds prefix.

### pickle
**Function**  
```python
pickle(cfg_path: Optional[str] = None, classes_path: Optional[str] = None, db_path: Optional[str] = None, output_path: Optional[str] = None, root_dir: str = '', max_per_class: Optional[int] = None, spec_group: Optional[str] = None) -> None
```
Convert database spectrograms to a pickle file for use in training.

This command extracts spectrograms from the training database and saves them in a pickle file
that can be efficiently loaded during model training. It can process all classes in the database
or specific classes specified by a CSV file.

Args:
    cfg_path (str, optional): Path to YAML file defining configuration overrides.
    classes_path (str, optional): Path to CSV file containing class names to include.
                                 If omitted, includes all classes in the database.
    db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    output_path (str, optional): Output pickle file path. Defaults to "data/training.pkl".
    max_per_class (int, optional): Maximum number of spectrograms to include per class.
    spec_group (str): Spectrogram group name to extract from. Defaults to 'default'.

### plot_db
**Function**  
```python
plot_db(cfg_path: Optional[str] = None, class_name: str = '', db_path: Optional[str] = None, ndims: bool = False, max_count: Optional[float] = None, output_path: str = '', prefix: Optional[str] = None, power: Optional[float] = 1.0, spec_group: Optional[str] = None)
```
Plot spectrograms from a training database for a specific class.

This command extracts spectrograms from the training database for a given class and
saves them as JPEG images. It can filter recordings by filename prefix and limit the
number of spectrograms plotted.

Args:
    cfg_path (str, optional): Path to YAML file defining configuration overrides.
    class_name (str): Name of the class to plot spectrograms for (e.g., "Common Yellowthroat").
    db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    ndims (bool): If True, do not show time and frequency dimensions on the spectrogram plots.
    max_count (int, optional): Maximum number of spectrograms to plot. If omitted, plots all available.
    output_path (str): Directory where spectrogram images will be saved.
    prefix (str, optional): Only include recordings that start with this filename prefix.
    power (float, optional): Raise spectrograms to this power for visualization. Lower values show more detail.
    spec_group (str, optional): Spectrogram group name to plot from. Defaults to "default".

### plot_dir
**Function**  
```python
plot_dir(cfg_path: Optional[str] = None, ndims: bool = False, input_path: str = '', output_path: str = '', all: bool = False, overlap: float = 0.0, power: float = 1.0)
```
Plot spectrograms for all audio recordings in a directory.

This command processes all audio files in a directory and generates spectrogram images.
It can either plot each recording as a single spectrogram or break recordings into
overlapping segments.

Args:
    cfg_path (str, optional): Path to YAML file defining configuration overrides.
    ndims (bool): If True, do not show time and frequency dimensions on the spectrogram plots.
    input_path (str): Directory containing audio recordings to process.
    output_path (str): Directory where spectrogram images will be saved.
    all (bool): If True, plot each recording as one spectrogram. If False, break into segments.
    overlap (float): Spectrogram overlap in seconds when breaking recordings into segments. Default is 0.
    power (float): Raise spectrograms to this power for visualization. Lower values show more detail. Default is 1.0.

### reextract
**Function**  
```python
reextract(cfg_path: Optional[str] = None, db_path: Optional[str] = None, class_name: Optional[str] = None, classes_path: Optional[str] = None, check: bool = False, spec_group: str = 'default')
```
Re-generate spectrograms from audio recordings and update the training database.

This command extracts spectrograms from audio recordings and imports them into the training database.
It can process all classes in the database or specific classes specified by name or CSV file.
If the specified spectrogram group already exists, it will be deleted and recreated.

In check mode, it only verifies that all required audio files are accessible without
updating the database.

Args:
    cfg_path (str, optional): Path to YAML file defining configuration overrides.
    db_path (str, optional): Path to the training database. Defaults to cfg.train.training_db.
    class_name (str, optional): Name of a specific class to reextract. If omitted, processes all classes.
    classes_path (str, optional): Path to CSV file listing classes to reextract. Alternative to class_name.
    check (bool): If True, only check that all recording paths are accessible without updating database.
    spec_group (str): Spectrogram group name for storing the extracted spectrograms. Defaults to 'default'.

### rpt_ann
**Function**  
```python
rpt_ann(annotations_path: str = '', output_path: str = '')
```
Summarize per-segment annotations from a test dataset.

This command reads annotation data from a CSV file and generates summary reports
showing the total duration of each class across all recordings and per-recording
breakdowns.

Args:
    annotations_path (str): Path to CSV file containing per-segment annotations.
    output_path (str): Directory where summary reports will be saved.

### rpt_db
**Function**  
```python
rpt_db(cfg_path: Optional[str] = None, db_path: Optional[str] = None, output_path: str = '')
```
Generate a comprehensive summary report of the training database.

This command analyzes the training database and generates detailed reports
about class distributions, spectrogram groups, and data organization.
The reports help understand the composition and quality of training data
and can be used for data management and quality control.

Args:
    cfg_path (str, optional): Path to YAML file defining configuration overrides.
    db_path (str, optional): Path to the training database. Defaults to cfg.train.train_db.
    output_path (str): Directory where database reports will be saved.

### rpt_epochs
**Function**  
```python
rpt_epochs(cfg_path: Optional[str] = '', input_path: str = '', annotations_path: str = '', output_path: str = '')
```
Given a checkpoint directory and a test, run every checkpoint against the test
and measure the macro-averaged ROC and AP scores, and then plot them.
This is useful to determine the number of training epochs needed.

Args:
    cfg_path (str, optional): Path to YAML file defining configuration overrides.
    input_path (str): Checkpoint directory generated by training.
    annotations_path (str): Path to CSV file containing ground truth annotations.
    output_path (str): Directory where the graph image will be saved.

### rpt_labels
**Function**  
```python
rpt_labels(label_dir: str = '', output_path: str = '', min_score: Optional[float] = None)
```
Summarize the output of an inference run.

This command processes inference results (from CSV files or Audacity labels)
and generates summary reports showing the total duration of detections
per class and per recording. It filters results by confidence threshold
and removes overlapping detections to provide clean statistics.

The reports help understand model performance and detection patterns
across different recordings and classes.

Args:
    label_dir (str): Directory containing inference output (CSV or Audacity labels).
    output_path (str): Directory where summary reports will be saved.
    min_score (float, optional): Ignore detections below this confidence threshold.

### rpt_test
**Function**  
```python
rpt_test(cfg_path: Optional[str] = None, granularity: str = 'segment', annotations_path: str = '', label_dir: str = '', output_path: str = '', recordings_path: Optional[str] = None, min_score: Optional[float] = None, precision: float = 0.95)
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
    cfg_path (str, optional): Path to YAML file defining configuration overrides.
    granularity (str): Evaluation granularity ("recording", "minute", or "segment"). Default is "segment".
    annotations_path (str): Path to CSV file containing ground truth annotations.
    label_dir (str): Directory containing model prediction labels (Audacity format).
    output_path (str): Directory where test reports will be saved.
    recordings_path (str, optional): Directory containing audio recordings. Defaults to annotations directory.
    min_score (float, optional): Provide detailed reports for this confidence threshold.
    precision (float): For recording granularity, report true positive seconds at this precision. Default is 0.95.

### search
**Function**  
```python
search(cfg_path: Optional[str] = None, db_path: Optional[str] = None, class_name: str = '', max_dist: float = 0.5, exp: float = 0.5, num_to_plot: int = 200, output_path: str = '', input_path: str = '', offset: float = 0.0, exclude_db: Optional[str] = None, class_name2: Optional[str] = None, spec_group: str = 'default')
```
Search a database for spectrograms similar to a specified one.

This command extracts a spectrogram from a given audio file at a specified offset,
then searches through a database of spectrograms to find the most similar ones
based on embedding similarity. Results are plotted and saved to the output directory.

Args:
    cfg_path (str): Path to YAML configuration file defining model settings.
    db_path (str): Path to the training database containing spectrograms to search.
    class_name (str): Name of the class/species to search within the database.
    max_dist (float): Maximum distance threshold. Results with distance greater than this are excluded.
    exp (float): Exponent to raise spectrograms to for visualization (shows background sounds).
    num_to_plot (int): Maximum number of similar spectrograms to plot and save.
    output_path (str): Directory where search results and plots will be saved.
    input_path (str): Path to the audio file containing the target spectrogram.
    offset (float): Time offset in seconds where the target spectrogram is extracted.
    exclude_db (str, optional): Path to an exclusion database. Spectrograms in this database are excluded from results.
    class_name2 (str, optional): Class name in the exclusion database. Defaults to the search class name.
    spec_group (str): Spectrogram group name in the database. Defaults to 'default'.

### train
**Function**  
```python
train(cfg_path: Optional[str] = None)
```
Train a bioacoustic recognition model using the specified configuration.

This command initiates the complete training pipeline for a bioacoustic model.
It loads training data from the database, configures the model architecture,
and runs the training process with the specified hyperparameters. The training
includes validation, checkpointing, and progress monitoring.

Training progress is displayed in real-time, and model checkpoints are saved
automatically. The final trained model can be used for inference and evaluation.

Args:
    cfg_path (str, optional): Path to YAML file defining configuration overrides.
                             If not specified, uses default configuration.

### tune
**Function**  
```python
tune(cfg_path: Optional[str] = None, param_path: Optional[str] = None, output_path: str = '', annotations_path: str = '', metric: str = 'macro_roc', recordings_path: str = '', train_log_path: str = '', num_trials: int = 0, num_runs: int = 1, extract: bool = False, skip_training: bool = False, classes_path: Optional[str] = None)
```
Find and print the best hyperparameter settings based on exhaustive or random search.

This command performs hyperparameter optimization by training models with different
parameter combinations and evaluating them using the specified metric. It can perform
either exhaustive search (testing all combinations) or random search (testing a
specified number of random combinations). To tune spectrogram settings, the --extract
CLI flag or API parameter specifies that new spectrograms will be extracted before training.
To tune inference settings, the --notrain CLI flag (skip_training API parameter) specifies
that training will be skipped.

The param_path specifies a YAML file that defines the parameters to be tuned, as described in the README.

Args:
    cfg_path (str, optional): Path to YAML file defining configuration overrides.
    param_path (str, optional): Path to YAML file defining hyperparameters to tune and their search space.
    output_path (str): Directory where reports will be saved.
    annotations_path (str): Path to CSV file containing ground truth annotations.
    metric (str): Metric used to compare runs. Options include various MAP and ROC metrics.
    recordings_path (str, optional): Directory containing audio recordings. Defaults to annotations directory.
    train_log_path (str, optional): Training log directory. Defaults to "logs/fold-0".
    num_trials (int): Number of random trials to run. If 0, performs exhaustive search.
    num_runs (int): Number of runs to average for each parameter combination. Default is 1.
    extract (bool): Extract new spectrograms before training, to tune spectrogram parameters.
    skip_training (bool): Iterate on inference only, using checkpoints from the last training run.
    classes_path (str, optional): Path to CSV containing class names for extract option. Default is all classes.

### wav2mp3
**Function**  
```python
wav2mp3(dir: str = '', sampling_rate: int = 32000)
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
    dir (str): Path to directory containing audio files to convert.
    sampling_rate (int): Output sampling rate in Hz. Default is 32000 Hz.

### xeno
**Function**  
```python
xeno(key: Optional[str] = None, name: str = '', output_dir: str = '', max_downloads: int = 500, ignore_licence: bool = False, scientific_name: bool = False, seen_only: bool = False)
```
Download bird song recordings from Xeno-Canto database.

This command uses the Xeno-Canto API v3 to search for and download audio recordings
of bird songs. The API requires authentication via an API key. Recordings are
downloaded as MP3 files and saved to the specified output directory.

To get an API key, register as a Xeno-Canto user and check your account page.
Then specify the key in the --key argument, or set the environment variable XCKEY=<key>.

Args:
    key (str): Xeno-Canto API key for authentication. Can also be set via XCKEY environment variable.
    output_dir (str): Directory where downloaded recordings will be saved.
    max_downloads (int): Maximum number of recordings to download. Default is 500.
    name (str): Species name to search for (common name or scientific name).
    ignore_licence (bool): If True, ignore license restrictions. By default, excludes BY-NC-ND licensed recordings.
    scientific_name (bool): If True, treat the name as a scientific name rather than common name.
    seen_only (bool): If True, only download recordings where the animal was seen (animal-seen=yes).

### youtube
**Function**  
```python
youtube(id: str = '', output_dir: str = '', sampling_rate: int = 32000) -> None
```
Download an audio recording from Youtube, given a Youtube ID.

Args:
    id (str): ID of the clip to download.
    output_dir (str): Directory where downloaded recordings will be saved.
    sampling_rate (float): Output sampling rate in Hz. Default is 32000.
