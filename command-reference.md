## Command Reference
| Command | Description |
| --- | --- |
| [britekit add-cat](#britekit-add-cat) | Add a category (class group) record to a database. |
| [britekit add-class](#britekit-add-class) | Add a class record to a database. |
| [britekit add-src](#britekit-add-src) | Add a source (e.g. 'Xeno-Canto') record to a database. |
| [britekit add-stype](#britekit-add-stype) | Add a soundtype record to a database. |
| [britekit analyze](#britekit-analyze) | Run inference. |
| [britekit audioset](#britekit-audioset) | Download recordings from Google Audioset. |
| [britekit ckpt-avg](#britekit-ckpt-avg) | Average the weights of several checkpoints. |
| [britekit ckpt-freeze](#britekit-ckpt-freeze) | Freeze the backbone weights of a checkpoint. |
| [britekit ckpt-onnx](#britekit-ckpt-onnx) | Convert a checkpoint to onnx format for use with openvino. |
| [britekit copy-samples](#britekit-copy-samples) | Copy packaged BriteKit sample YAML/CSV files to a destination directory. |
| [britekit del-cat](#britekit-del-cat) | Delete a category (class group) and its classes from a database. |
| [britekit del-class](#britekit-del-class) | Delete a class and associated records from a database. |
| [britekit del-rec](#britekit-del-rec) | Delete a recording and associated records from a database. |
| [britekit del-sgroup](#britekit-del-sgroup) | Delete a spectrogram group from the database. |
| [britekit del-spec](#britekit-del-spec) | Delete spectrograms that match given images. |
| [britekit del-src](#britekit-del-src) | Delete a recording source and associated records from a database. |
| [britekit del-stype](#britekit-del-stype) | Delete a sound type from a database. |
| [britekit embed](#britekit-embed) | Insert spectrogram embeddings into database. |
| [britekit extract-all](#britekit-extract-all) | Insert all spectrograms from recordings into database. |
| [britekit extract-by-image](#britekit-extract-by-image) | Insert spectrograms that correspond to images. |
| [britekit find-dup](#britekit-find-dup) | Find and optionally delete duplicate recordings in a database. |
| [britekit find-lr](#britekit-find-lr) | Suggest a learning rate. |
| [britekit inat](#britekit-inat) | Download recordings from iNaturalist. |
| [britekit pickle](#britekit-pickle) | Convert database records to a pickle file for use in training. |
| [britekit plot-db](#britekit-plot-db) | Plot spectrograms from a database. |
| [britekit plot-dir](#britekit-plot-dir) | Plot spectrograms from a directory of recordings. |
| [britekit plot-rec](#britekit-plot-rec) | Plot spectrograms from a specific recording. |
| [britekit reextract](#britekit-reextract) | Re-generate the spectrograms in a database, and add them to the database. |
| [britekit rpt-ann](#britekit-rpt-ann) | Summarize annotations in a per-segment test. |
| [britekit rpt-cal](#britekit-rpt-cal) | Calibrate an ensemble based on per-segment test results. |
| [britekit rpt-db](#britekit-rpt-db) | Generate a database summary report. |
| [britekit rpt-labels](#britekit-rpt-labels) | Summarize the output of an inference run. |
| [britekit rpt-test](#britekit-rpt-test) | Generate metrics and reports from test results. |
| [britekit search](#britekit-search) | Search a database for spectrograms similar to one given. |
| [britekit train](#britekit-train) | Run training. |
| [britekit tune](#britekit-tune) | Tune hyperparameters using exhaustive or random search. |
| [britekit wav2mp3](#britekit-wav2mp3) | Convert uncompressed audio or flac to mp3. |
| [britekit xeno](#britekit-xeno) | Download recordings from Xeno-Canto. |

### britekit add-cat
```
Usage: cli add-cat [OPTIONS]

  Add a category (class group) record to the training database.

  Categories are used to group related classes together in the database. For
  example, you might have categories like "Birds", "Mammals", or "Insects" that
  contain multiple related species classes.

Options:
  -d, --db TEXT  Path to the database.
  --name TEXT    Category name  [required]
  --help         Show this message and exit.
```
### britekit add-class
```
Usage: cli add-class [OPTIONS]

  Add a class record to the training database.

  Classes represent the target species or sound categories for training and
  inference. Each class must belong to a category and can have both primary and
  alternate names/codes. This is typically used to add new species or sound
  types to the training database.

Options:
  -d, --db TEXT    Path to the database.
  --cat TEXT       Category name
  --name TEXT      Class name  [required]
  --code TEXT      Class code  [required]
  --alt_name TEXT  Class alternate name
  --alt_code TEXT  Class alternate code
  --help           Show this message and exit.
```
### britekit add-src
```
Usage: cli add-src [OPTIONS]

  Add a source record to the training database.

  Sources track where audio recordings originated from, such as "Xeno-Canto",
  "Macaulay Library", "iNaturalist", or custom field recordings. This helps
  maintain provenance and can be useful for data quality analysis.

Options:
  -d, --db TEXT  Path to the database.
  --name TEXT    Source name  [required]
  --help         Show this message and exit.
```
### britekit add-stype
```
Usage: cli add-stype [OPTIONS]

  Add a sound type record to the training database.

  Sound types describe the nature of the audio content, such as "Song", "Call",
  "Alarm", "Drumming", etc. This helps categorize different types of
  vocalizations or sounds produced by the same species.

Options:
  -d, --db TEXT  Path to the database.
  --name TEXT    Soundtype name  [required]
  --help         Show this message and exit.
```
### britekit analyze
```
Usage: cli analyze [OPTIONS]

  Run inference on audio recordings to detect and classify sounds.

  This command processes audio files or directories and generates predictions
  using a trained model or ensemble. The output can be saved as Audacity labels,
  CSV files, or both.

Options:
  -c, --cfg PATH          Path to YAML file defining config overrides.
  -i, --input PATH        Path to input directory or recording.
  -o, --output DIRECTORY  Path to output directory (optional, defaults to input
                          directory).
  -r, --rtype TEXT        Output format type. Options are "audacity", "csv", or
                          "both". Default="both".
  -m, --min_score FLOAT   Threshold, so predictions lower than this value are
                          excluded.
  --threads INTEGER       Number of threads (optional, default = 3)
  --overlap FLOAT         Number of threads (optional, default = 3)
  --seg FLOAT             Optional segment length in seconds. If specified,
                          labels are fixed-length. Otherwise they are variable-
                          length.
  --help                  Show this message and exit.
```
### britekit audioset
```
Usage: cli audioset [OPTIONS]

  Download audio recordings from Google AudioSet.

  This command downloads audio clips from Google AudioSet, a large-scale dataset
  of audio events. You can either download a curated set of recordings or search
  for a specific audio class. When using --rpt flag with a class name, it
  generates a report on associated secondary classes instead of downloading
  recordings.

  Most AudioSet clips contain multiple classes (e.g., "train", "wind",
  "speech"). The report shows which other classes commonly co-occur with the
  specified class.

Options:
  --name TEXT         Class name.
  --curated FILE      Path to CSV with curated list of clips.
  --output DIRECTORY  Output directory.  [required]
  --max INTEGER       Maximum number of recordings to download. Default = 500.
  --sr INTEGER        Output sampling rate (default = 32000).
  --skip INTEGER      Skip this many initial recordings (default = 0).
  --rpt               Report on secondary classes associated with the specified
                      class.
  --help              Show this message and exit.
```
### britekit ckpt-avg
```
Usage: cli ckpt-avg [OPTIONS]

  Average the weights of multiple model checkpoints to create an ensemble
  checkpoint.

  This command loads multiple checkpoint files from a directory and creates a
  new checkpoint with averaged weights.

Options:
  -i, --input DIRECTORY  Directory containing checkpoints to average  [required]
  -o, --output FILE      Optional path to output checkpoint. Default is
                         weights.ckpt in the input directory
  --help                 Show this message and exit.
```
### britekit ckpt-freeze
```
Usage: cli ckpt-freeze [OPTIONS]

  Freeze the backbone weights of a checkpoint to reduce file size and improve
  inference speed.

  This command loads a PyTorch checkpoint and freezes the backbone weights,
  which removes training-specific information like gradients and optimizer
  states. This significantly reduces the checkpoint file size and can improve
  inference performance.

  The original checkpoint is preserved with a ".original" extension, and the
  frozen version replaces the original file. Frozen checkpoints are optimized
  for deployment and inference rather than continued training.

Options:
  -i, --input FILE  Path to checkpoint to freeze  [required]
  --help            Show this message and exit.
```
### britekit ckpt-onnx
```
Usage: cli ckpt-onnx [OPTIONS]

  Convert a PyTorch checkpoint to ONNX format for deployment with OpenVINO.

  This command converts a trained PyTorch model checkpoint to ONNX (Open Neural
  Network Exchange) format, which enables deployment using Intel's OpenVINO
  toolkit. ONNX format allows for optimized inference on CPU.

  The conversion process creates a new ONNX file with the same base name as the
  input checkpoint.

Options:
  -c, --cfg PATH    Path to YAML file defining config overrides.
  -i, --input FILE  Path to checkpoint to convert to ONNX format  [required]
  --help            Show this message and exit.
```
### britekit copy-samples
```
Usage: cli copy-samples [OPTIONS]

  Copy packaged BriteKit sample YAML/CSV files to a destination directory.

  This command copies files from the built-in `britekit.samples` package (kept
  alongside the library code) into a folder you specify. You can filter by
  subpaths using a simple glob pattern and optionally do a dry-run list.

Options:
  --dest DIRECTORY              Directory to copy sample files into.  [required]
  --pattern TEXT                Glob-like filter relative to 'samples/' (e.g.
                                'full/*.yaml' or 'data/*.csv').  [default: *]
  --list                        List files that would be copied and exit.
  --overwrite / --no-overwrite  Overwrite existing files in DEST.  [default: no-
                                overwrite]
  --help                        Show this message and exit.
```
### britekit del-cat
```
Usage: cli del-cat [OPTIONS]

  Delete a category and all its associated data from the training database.

  This command performs a cascading delete that removes the category, all its
  classes, all recordings belonging to those classes, and all spectrograms from
  those recordings. This is a destructive operation that cannot be undone.

Options:
  -d, --db TEXT  Path to the training database.
  --name TEXT    Category name.  [required]
  --help         Show this message and exit.
```
### britekit del-class
```
Usage: cli del-class [OPTIONS]

  Delete a class and all its associated data from the training database.

  This command removes the class, all recordings belonging to that class, and
  all spectrograms from those recordings. This is a destructive operation that
  cannot be undone and will affect any training data associated with this class.

Options:
  -d, --db TEXT  Path to the training database.
  --name TEXT    Class name.  [required]
  --help         Show this message and exit.
```
### britekit del-rec
```
Usage: cli del-rec [OPTIONS]

  Delete a recording and all its spectrograms from the training database.

  This command removes a specific audio recording and all spectrograms that were
  extracted from it.

Options:
  -d, --db TEXT  Path to the training database.
  --name TEXT    Recording file name.  [required]
  --help         Show this message and exit.
```
### britekit del-sgroup
```
Usage: cli del-sgroup [OPTIONS]

  Delete a spectrogram group and all its spectrogram values from the training
  database.

  Spectrogram groups organize spectrograms by processing parameters or
  extraction method. This command removes the entire group and all spectrograms
  within it.

Options:
  -d, --db TEXT  Path to the training database.
  --name TEXT    Spec group name.  [required]
  --help         Show this message and exit.
```
### britekit del-spec
```
Usage: cli del-spec [OPTIONS]

  Delete spectrograms that correspond to images in a given directory.

  This command parses image filenames to identify and delete corresponding
  spectrograms from the database. Images are typically generated by the plot-db
  or search commands, and their filenames contain the recording name and time
  offset.

  This is useful for removal of spectrograms based on visual inspection of
  plots, allowing you to remove low-quality or incorrectly labeled spectrograms.

Options:
  -d, --db TEXT  Path to the training database.
  --class TEXT   Class name.  [required]
  --dir TEXT     Path to directory containing images.  [required]
  --help         Show this message and exit.
```
### britekit del-src
```
Usage: cli del-src [OPTIONS]

  Delete a recording source and all its associated data from the training
  database.

  This command performs a cascading delete that removes the source, all
  recordings from that source, and all spectrograms from those recordings. This
  is useful for removing entire datasets from a specific source (e.g., removing
  all Xeno-Canto data).

Options:
  -d, --db TEXT  Path to the training database.
  --name TEXT    Source name.  [required]
  --help         Show this message and exit.
```
### britekit del-stype
```
Usage: cli del-stype [OPTIONS]

  Delete a sound type from the training database.

  This command removes a sound type definition but preserves the spectrograms
  that were labeled with this sound type. The spectrograms will have their
  soundtype_id field set to null, effectively removing the sound type
  classification while keeping the audio data.

Options:
  -d, --db TEXT  Path to the training database.
  --name TEXT    Sound type name.  [required]
  --help         Show this message and exit.
```
### britekit embed
```
Usage: cli embed [OPTIONS]

  Generate embeddings for spectrograms and insert them into the database.

  This command uses a trained model to generate embeddings (feature vectors) for
  spectrograms in the training database. These embeddings can be used for
  similarity search and other downstream tasks. The embeddings are compressed
  and stored in the database.

Options:
  -c, --cfg PATH  Path to YAML file defining config overrides.
  -d, --db TEXT   Path to the database.
  --name TEXT     Class name
  --sgroup TEXT   Spectrogram group name. Defaults to 'default'.
  --help          Show this message and exit.
```
### britekit extract-all
```
Usage: cli extract-all [OPTIONS]

  Extract all spectrograms from audio recordings and insert them into the
  training database.

  This command processes all audio files in a directory and extracts
  spectrograms using sliding windows with optional overlap. The spectrograms are
  then inserted into the training database for use in model training. If the
  specified class doesn't exist, it will be automatically created.

Options:
  -c, --cfg PATH   Path to YAML file defining config overrides.
  -d, --db TEXT    Path to the training database.
  --cat TEXT       Category name, e.g. 'bird' for when new class is added.
                   Defaults to 'default'.
  --code TEXT      Class code for when new class is added.
  --name TEXT      Class name.  [required]
  --dir DIRECTORY  Path to directory containing recordings..  [required]
  --overlap FLOAT  Spectrogram overlap in seconds. Defaults to value in the
                   config file.
  --src TEXT       Source name for inserted recordings. Defaults to 'default'.
  --sgroup TEXT    Spectrogram group name. Defaults to 'default'.
  --help           Show this message and exit.
```
### britekit extract-by-image
```
Usage: cli extract-by-image [OPTIONS]

  Extract spectrograms that correspond to existing spectrogram images.

  This command parses spectrogram image filenames to identify the corresponding
  audio segments and extracts those specific spectrograms from the original
  recordings. This is useful when you have pre-selected spectrograms (e.g., from
  manual review or search results) and want to extract only those specific
  segments.

  The images contain metadata in their filenames (recording name and time
  offset) that allows the command to locate and extract the corresponding audio
  segments.

Options:
  -c, --cfg PATH        Path to YAML file defining config overrides.
  -d, --db TEXT         Path to the training database.
  --cat TEXT            Category name, e.g. 'bird' for when new class is added.
                        Defaults to 'default'.
  --code TEXT           Class code for when new class is added.
  --name TEXT           Class name.  [required]
  --rec-dir DIRECTORY   Path to directory containing recordings.  [required]
  --spec-dir DIRECTORY  Path to directory containing spectrogram images.
                        [required]
  --dest-dir DIRECTORY  Copy used recordings to this directory if specified.
  --src TEXT            Source name for inserted recordings. Defaults to
                        'default'.
  --sgroup TEXT         Spectrogram group name. Defaults to 'default'.
  --help                Show this message and exit.
```
### britekit find-dup
```
Usage: cli find-dup [OPTIONS]

  Find and optionally delete duplicate recordings in the training database.

  This command scans the database for recordings of the same class that appear
  to be duplicates. It uses a two-stage detection approach: 1. Compare recording
  durations (within 0.1 seconds tolerance) 2. Compare spectrogram embeddings of
  the first few spectrograms (within 0.02 cosine distance)

  Duplicates are identified by comparing the first 3 spectrogram embeddings from
  each recording using cosine distance.

Options:
  -c, --cfg PATH  Path to YAML file defining config overrides.
  -d, --db FILE   Path to the database. Defaults to value of
                  cfg.train.training_db.
  --name TEXT     Class name  [required]
  --del           If specified, remove duplicate recordings from the database.
  --sgroup TEXT   Spectrogram group name. Defaults to 'default'.
  --help          Show this message and exit.
```
### britekit find-lr
```
Usage: cli find-lr [OPTIONS]

  Find an optimal learning rate for model training using the learning rate
  finder.

  This command runs a learning rate finder that tests a range of learning rates
  on a small number of training batches to determine the optimal learning rate.
  It generates a plot showing loss vs. learning rate and suggests the best rate
  based on the steepest negative gradient in the loss curve.

  The suggested learning rate helps ensure stable and efficient training by
  avoiding rates that are too high (causing instability) or too low (slow
  convergence).

Options:
  -c, --cfg PATH             Path to YAML file defining config overrides.
  -n, --num-batches INTEGER  Number of batches to analyze
  --help                     Show this message and exit.
```
### britekit inat
```
Usage: cli inat [OPTIONS]

  Download audio recordings from iNaturalist observations.

  This command searches iNaturalist for observations of a specified species that
  contain audio recordings. It downloads the audio files and creates a CSV file
  mapping the downloaded files to their iNaturalist observation URLs for
  reference.

  Only observations with "research grade" quality are downloaded (excluding
  "needs_id"). The command respects the maximum download limit and can
  optionally add filename prefixes.

Options:
  --output DIRECTORY  Output directory.  [required]
  --max INTEGER       Maximum number of recordings to download. Default = 500.
  --noprefix          By default, filenames use an 'N' prefix and recording
                      number. Specify this flag to skip the prefix.
  --name TEXT         Species name.  [required]
  --help              Show this message and exit.
```
### britekit pickle
```
Usage: cli pickle [OPTIONS]

  Convert database spectrograms to a pickle file for use in training.

  This command extracts spectrograms from the training database and saves them
  in a pickle file that can be efficiently loaded during model training. It can
  process all classes in the database or specific classes specified by a CSV
  file.

Options:
  -c, --cfg PATH     Path to YAML file defining config overrides.
  --classes TEXT     Path to CSV containing class names to pickle (optional).
                     Default is all in database.
  -d, --db TEXT      Path to the training database.
  -o, --output TEXT  Output file path. Default is "data/training.pkl".
  -m, --max INTEGER  Maximum spectrograms per class.
  --sgroup TEXT      Spectrogram group name. Defaults to 'default'.
  --help             Show this message and exit.
```
### britekit plot-db
```
Usage: cli plot-db [OPTIONS]

  Plot spectrograms from a training database for a specific class.

  This command extracts spectrograms from the training database for a given
  class and saves them as JPEG images. It can filter recordings by filename
  prefix and limit the number of spectrograms plotted.

Options:
  -c, --cfg PATH          Path to YAML file defining config overrides.
  --name TEXT             Plot spectrograms for this class.  [required]
  -d, --db TEXT           Path to the training database.
  --dims                  If specified, show seconds on x-axis and frequencies
                          on y-axis.
  --max INTEGER           Max number of spectrograms to plot.
  -o, --output DIRECTORY  Path to output directory.  [required]
  --prefix TEXT           Only include recordings that start with this prefix.
  --power FLOAT           Raise spectrograms to this power. Lower values show
                          more detail.
  --sgroup TEXT           Spectrogram group name. Defaults to 'default'.
  --help                  Show this message and exit.
```
### britekit plot-dir
```
Usage: cli plot-dir [OPTIONS]

  Plot spectrograms for all audio recordings in a directory.

  This command processes all audio files in a directory and generates
  spectrogram images. It can either plot each recording as a single spectrogram
  or break recordings into overlapping segments.

Options:
  -c, --cfg PATH          Path to YAML file defining config overrides.
  --dims                  If specified, show seconds on x-axis and frequencies
                          on y-axis.
  -i, --input DIRECTORY   Path to input directory.  [required]
  -o, --output DIRECTORY  Path to output directory.  [required]
  --all                   If specified, plot whole recordings in one spectrogram
                          each. Otherwise break them up into segments.
  --overlap FLOAT         Spectrogram overlap in seconds. Default = 0.
  --power FLOAT           Raise spectrograms to this power. Lower values show
                          more detail.
  --help                  Show this message and exit.
```
### britekit plot-rec
```
Usage: cli plot-rec [OPTIONS]

  Plot spectrograms for a specific audio recording.

  This command processes a single audio file and generates spectrogram images.
  It can either plot the entire recording as one spectrogram or break it into
  overlapping segments.

Options:
  -c, --cfg PATH          Path to YAML file defining config overrides.
  --dims                  If specified, show seconds on x-axis and frequencies
                          on y-axis.
  -i, --input FILE        Path to input directory.  [required]
  -o, --output DIRECTORY  Path to output directory.  [required]
  --all                   If specified, plot whole recordings in one spectrogram
                          each. Otherwise break them up into segments.
  --overlap FLOAT         Spectrogram overlap in seconds. Default = 0.
  --power FLOAT           Raise spectrograms to this power. Lower values show
                          more detail.
  --help                  Show this message and exit.
```
### britekit reextract
```
Usage: cli reextract [OPTIONS]

  Re-generate spectrograms from audio recordings and update the training
  database.

  This command extracts spectrograms from audio recordings and imports them into
  the training database. It can process all classes in the database or specific
  classes specified by name or CSV file. If the specified spectrogram group
  already exists, it will be deleted and recreated.

  In check mode, it only verifies that all required audio files are accessible
  without updating the database.

Options:
  -c, --cfg PATH  Path to YAML file defining config overrides.
  -d, --db FILE   Path to the database. Defaults to value of
                  cfg.train.training_db.
  --name TEXT     Optional class name. If this and --csv are omitted, do all
                  classes.
  --classes FILE  Path to CSV listing classes to reextract. Alternative to
                  --name. If this and --name are omitted, do all classes.
  --check         If specified, just check if all specified recordings are
                  accessible and do not update the database.
  --sgroup TEXT   Spectrogram group name. Defaults to 'default'.
  --help          Show this message and exit.
```
### britekit rpt-ann
```
Usage: cli rpt-ann [OPTIONS]

  Summarize per-segment annotations from a test dataset.

  This command reads annotation data from a CSV file and generates summary
  reports showing the total duration of each class across all recordings and
  per-recording breakdowns.

Options:
  -a, --annotations FILE  Path to CSV file containing annotations or ground
                          truth).  [required]
  -o, --output DIRECTORY  Path to output directory.  [required]
  --help                  Show this message and exit.
```
### britekit rpt-cal
```
Usage: cli rpt-cal [OPTIONS]

  Calibrate model predictions using per-segment test results.

  This command generates calibration plots and analysis to assess how well model
  prediction scores align with actual probabilities. It compares predicted
  scores against ground truth annotations to determine if the model is
  overconfident or underconfident in its predictions.

  The calibration process helps improve model reliability by adjusting
  prediction scores to better reflect true probabilities.

Options:
  -c, --cfg PATH              Path to YAML file defining config overrides.
  -a, --annotations FILE      Path to CSV file containing annotations or ground
                              truth).  [required]
  -l, --labels TEXT           Directory containing Audacity labels. If a
                              subdirectory of recordings directory, only the
                              subdirectory name is needed.  [required]
  -o, --output DIRECTORY      Path to output directory.  [required]
  --classes FILE              Path to CSV listing classes included in training.
                              [required]
  -r, --recordings DIRECTORY  Recordings directory. Default is directory
                              containing annotations file.
  --cutoff FLOAT              When calibrating, ignore predictions below this
                              (default = .4)
  --coef FLOAT                Use this coefficient in the calibration plot.
  --inter FLOAT               Use this intercept in the calibration plot.
  --help                      Show this message and exit.
```
### britekit rpt-db
```
Usage: cli rpt-db [OPTIONS]

  Generate a comprehensive summary report of the training database.

  This command analyzes the training database and generates detailed reports
  about class distributions, spectrogram groups, and data organization. The
  reports help understand the composition and quality of training data and can
  be used for data management and quality control.

Options:
  -c, --cfg PATH          Path to YAML file defining config overrides.
  -d, --db TEXT           Path to the training database.
  -o, --output DIRECTORY  Path to output directory.  [required]
  --help                  Show this message and exit.
```
### britekit rpt-labels
```
Usage: cli rpt-labels [OPTIONS]

  Summarize the output of an inference run.

  This command processes inference results (from CSV files or Audacity labels)
  and generates summary reports showing the total duration of detections per
  class and per recording. It filters results by confidence threshold and
  removes overlapping detections to provide clean statistics.

  The reports help understand model performance and detection patterns across
  different recordings and classes.

Options:
  -l, --labels TEXT       Directory containing inference output (CSV file or
                          Audacity label files).  [required]
  -o, --output DIRECTORY  Directory to store output reports.  [required]
  -m, --min_score FLOAT   Ignore scores below this threshold.
  --help                  Show this message and exit.
```
### britekit rpt-test
```
Usage: cli rpt-test [OPTIONS]

  Generate comprehensive test metrics and reports comparing model predictions to
  ground truth.

  This command evaluates model performance by comparing inference results
  against ground truth annotations. It supports three granularity levels: -
  "recording": Evaluate at the recording level (presence/absence) - "minute":
  Evaluate at the minute level (presence/absence per minute) - "segment":
  Evaluate at the segment level (detailed temporal alignment)

  The command generates detailed performance metrics including precision,
  recall, F1 scores, and various visualization plots to help understand model
  behavior.

Options:
  -c, --cfg PATH              Path to YAML file defining config overrides.
  -g, --granularity TEXT      Test annotation and reporting granularity
                              ("recording", "minute" or "segment"). Default =
                              "segment".
  -a, --annotations FILE      Path to CSV file containing annotations or ground
                              truth).  [required]
  -l, --labels TEXT           Directory containing Audacity labels. If a
                              subdirectory of recordings directory, only the
                              subdirectory name is needed.  [required]
  -o, --output DIRECTORY      Path to output directory.  [required]
  --classes FILE              Path to CSV listing classes included in training.
                              [required]
  -r, --recordings DIRECTORY  Recordings directory. Default is directory
                              containing annotations file.
  -m, --min_score FLOAT       Provide detailed reports for this threshold.
  --precision FLOAT           For granularity=recording, report TP seconds at
                              this precision (default=.95).
  --help                      Show this message and exit.
```
### britekit search
```
Usage: cli search [OPTIONS]

  Search a database for spectrograms similar to a specified one.

  This command extracts a spectrogram from a given audio file at a specified
  offset, then searches through a database of spectrograms to find the most
  similar ones based on embedding similarity. Results are plotted and saved to
  the output directory.

Options:
  -c, --cfg PATH          Path to YAML file defining config overrides.
  -d, --db FILE           Path to the database. Defaults to value of
                          cfg.train.training_db.
  --name TEXT             Class name  [required]
  --dist FLOAT            Exclude results with distance greater than this.
                          Default = .5.
  --exp FLOAT             Raise spectrograms to this exponent to show background
                          sounds. Default = .5.
  --num INTEGER           Only plot up to this many spectrograms. Default = 200.
  -o, --output DIRECTORY  Path to output directory.  [required]
  -i, --input FILE        Path to recording containing spectrogram to search
                          for.  [required]
  --offset FLOAT          Offset in seconds of spectrogram to search for.
                          Default = 0.
  -x, --exclude FILE      If specified, exclude spectrograms that exist in this
                          database.
  --name2 TEXT            If --exclude is specified, this is class name in
                          exclude database. Default is the search class name.
  --sgroup TEXT           Spectrogram group name. Defaults to 'default'.
  --help                  Show this message and exit.
```
### britekit train
```
Usage: cli train [OPTIONS]

  Train a bioacoustic recognition model using the specified configuration.

  This command initiates the complete training pipeline for a bioacoustic model.
  It loads training data from the database, configures the model architecture,
  and runs the training process with the specified hyperparameters. The training
  includes validation, checkpointing, and progress monitoring.

  Training progress is displayed in real-time, and model checkpoints are saved
  automatically. The final trained model can be used for inference and
  evaluation.

Options:
  -c, --cfg PATH  Path to YAML file defining config overrides.
  --help          Show this message and exit.
```
### britekit tune
```
Usage: cli tune [OPTIONS]

  Find and print the best hyperparameter settings based on exhaustive or random
  search.

  This command performs hyperparameter optimization by training models with
  different parameter combinations and evaluating them using the specified
  metric. It can perform either exhaustive search (testing all combinations) or
  random search (testing a specified number of random combinations).

  The param_path specifies a YAML file that contains a sequence of parameters
  such as:

  - name: prob_simple_merge   type: float   bounds:   - 0.36   - 0.42   step:
  0.02

  The name defines a hyperparameter to tune, with given type, bounds and step
  size.

Options:
  -c, --cfg PATH                  Path to YAML file defining config overrides.
  -p, --param PATH                Path to YAML file defining hyperparameters to
                                  tune.  [required]
  -a, --annotations FILE          Path to CSV file containing annotations or
                                  ground truth).  [required]
  --classes FILE                  Path to CSV listing classes included in
                                  training.  [required]
  -m, --metric [macro_map|micro_map_annotated|micro_map_trained|combined_map_annotated|combined_map_trained|macro_roc|micro_roc_annotated|micro_roc_trained|combined_roc_annotated|combined_roc_trained]
                                  Metric used to compare runs.
  -r, --recordings DIRECTORY      Recordings directory. Default is directory
                                  containing annotations file.
  --log DIRECTORY                 Training log directory.
  --trials INTEGER                If specified, run this many random trials.
                                  Otherwise do an exhaustive search.
  --runs INTEGER                  Use the average score of this many runs in
                                  each case. Default = 1.
  --help                          Show this message and exit.
```
### britekit wav2mp3
```
Usage: cli wav2mp3 [OPTIONS]

  Convert uncompressed audio files to MP3 format and replace the originals.

  This command processes all uncompressed audio files in a directory and
  converts them to MP3 format using FFmpeg. Supported input formats include
  FLAC, WAV, WMA, AIFF, and other uncompressed audio formats. After successful
  conversion, the original files are deleted to save disk space.

  The conversion uses a 192k bitrate for good quality while maintaining
  reasonable file sizes. This is useful for standardizing audio formats and
  reducing storage requirements for large audio datasets.

Options:
  --dir DIRECTORY  Path to directory containing recordings.  [required]
  --sr INTEGER     Output sampling rate (default = 32000).
  --help           Show this message and exit.
```
### britekit xeno
```
Usage: cli xeno [OPTIONS]

  Download bird song recordings from Xeno-Canto database.

  This command uses the Xeno-Canto API v3 to search for and download audio
  recordings of bird songs. The API requires authentication via an API key.
  Recordings are downloaded as MP3 files and saved to the specified output
  directory.

  To get an API key, register as a Xeno-Canto user and check your account page.
  Then specify the key in the --key argument, or set the environment variable
  XCKEY=<key>.

Options:
  --key TEXT          Xeno-Canto API key.
  --output DIRECTORY  Output directory.  [required]
  --max INTEGER       Maximum number of recordings to download. Default = 500.
  --name TEXT         Species name.  [required]
  --nolic             Specify this flag to ignore the licence. By default,
                      exclude if licence is BY-NC-ND.
  --sci               Specify this flag when using a scientific name rather than
                      a common name.
  --seen              Specify this flag to download only if animal-seen=yes.
  --help              Show this message and exit.
```
