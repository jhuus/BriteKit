# BriteKit

-----

## Getting Started

- [Introduction](#introduction)
- [License](#license)
- [Installation](#installation)
- [Configuration](#configuration)
- [Downloading Recordings](#downloading-recordings)
- [Managing Training Data](#managing-training-data)
- [Training](#training)
- [Testing](#testing)
- [Tuning](#tuning)
- [Ensembling](#ensembling)
- [Calibrating](#calibrating)

## More Information

- [Spectrograms](#spectrograms)
- [Backbones and Classifier Heads](#backbones-and-classifier-heads)
- [Metrics (PR-AUC and ROC-AUC)](#metrics-pr-auc-and-roc-auc)
- [Data Augmentation](#data-augmentation)
- [Development Environment](#development-environment)

## Reference Guides

- [Command Reference](https://github.com/jhuus/BriteKit/blob/master/command-reference.md)
- [Command API Reference](https://github.com/jhuus/BriteKit/blob/master/command-api-reference.md)
- [General API Reference](https://github.com/jhuus/BriteKit/blob/master/api-reference.md)
- [Configuration Reference](https://github.com/jhuus/BriteKit/blob/master/config-reference.md)

# Getting Started

-----

## Introduction
BriteKit (Bioacoustic Recognizer Technology Kit) is a Python package that facilitates the development of bioacoustic recognizers using deep learning.
It provides a command-line interface (CLI) as well as a Python API, to support functions such as:
- downloading recordings from Xeno-Canto, iNaturalist, and YouTube (optionally using Google Audioset metadata)
- managing training data in a SQLite database
- training models
- testing, tuning and calibrating models
- reporting
- deployment and inference

To view a list of BriteKit commands, type `britekit --help`. You can also get help for individual commands, e.g. `britekit train --help` describes the `train` command.
When accessing BriteKit from Python, the `britekit.commands` namespace contains a function for each command, as documented [here](command-api-reference.md).
The classes used by the commands can also be accessed, and are documented [here](api-reference.md).
## License
BriteKit is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
## Installation
It is best to install BriteKit in a virtual environment, such as a [Python venv](https://docs.python.org/3/library/venv.html). Once you have that set up, install the BriteKit package using pip:
```console
pip install britekit
```
In Windows environments, you then need to uninstall and reinstall PyTorch:
```
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
Note that cu126 refers to CUDA 12.6.\
Once BriteKit is installed, initialize a working environment using the `init` command:
```console
britekit init --dest=<directory path>
```
This creates the directories needed and installs sample files. If you omit `--dest`, it will create
directories under the current working directory.
## Configuration
Configuration parameters are documented [here](config-reference.md). After running `britekit init`, the file `yaml/base_config.yaml` contains all parameters in YAML format.
Most CLI commands have a `--config` argument that allows you to specify the path to a YAML file that overrides selected parameters. For example, when running the `train` command,
you could provide a YAML file containing the following:
```
train:
  model_type: "effnet.4"
  learning_rate: .002
  drop_rate: 0.1
  num_epochs: 20
```
This overrides the default values for model_type, learning_rate, drop_rate and num_epochs. When using the API, you can update configuration parameters like this:
```
import britekit as bk
cfg = bk.get_config()
cfg.train.model_type = "effnet.4"
```
## Downloading Recordings
The `inat`, `xeno` and `youtube` commands make it easy to download recordings from Xeno-Canto, iNaturalist and YouTube. For iNaturalist it is important to provide the scientific name. For example, to download recordings of the American Green Frog (lithobates clamitans), type:
```
britekit inat --name "lithobates clamitans" --output <output-path>
```
For Xeno-Canto, use `--name` for the common name or `--sci` for the scientific name. For YouTube, specify the ID of the corresponding video. For example, specify `--id K_EsxukdNXM` to download the audio from https://www.youtube.com/watch?v=K_EsxukdNXM.

BriteKit also supports downloads using [Google Audioset](https://research.google.com/audioset/), which is metadata that classifies sounds in YouTube videos. Audioset was released in March 2017, so any videos uploaded later than that are not included. Also, some videos that are tagged in Audioset are no longer available. Type `britekit audioset --help` for more information.
## Managing Training Data
Once you have a collection of recordings, the steps to prepare it for training are:
1. Extract spectrograms from recordings and insert them into the training database.
2. Curate the training spectrograms.
3. Create a pickle file from the training data.
Then provide the path to the pickle file when running training.

Suppose we have a folder called `recordings/cow`. To generate spectrograms and insert them into the training database, we could type `britekit extract-all --name Cow --dir recordings/cow`. This will create a SQLite database in `data/training.db` and populate it with spectrograms using the default configuration.
To browse the database, you can use [DB Browser for SQLite](https://sqlitebrowser.org/), or a similar application.
That will reveal the following tables:
- Class: classes that the recognizer will be trained to identify, e.g. American Robin
- Category: categories such as Bird, Mammal or Amphibian
- Source: sources of recordings, e.g. Xeno-Canto or iNaturalist.
- Recording: individual recordings
- Segment: fixed-length sections of recordings
- SpecGroup: groups of spectrograms that share spectrogram parameters
- SpecValue: spectrograms, each referencing a Segment and SpecGroup
- SegmentClass: associations between Segment and Class, to identify the classes that occur in a segment

There are commands to add or delete database records, e.g. `add-cat` and `del-cat` to add or delete a category record. In addition, specifying the `--cat` argument with the `extract-all` or `extract-by-image` commands will add the required category record if it does not exist. You can plot database spectrograms using `plot-db`, or plot spectrograms for recordings using `plot-rec` or `plot-dir`. Once you have a folder of spectrogram images, you can manually delete or copy some of them. The `extract-by-image` command will then extract only the spectrograms corresponding to the given images. Similarly, the `del-spec` command will delete spectrograms corresponding to the images in a directory.

It is important to tune spectrogram parameters such as height, width, maximum/minimum frequency and window length for your specific application. This is discussed more in the tuning section below, but for now be aware that you can set specific parameters in a YAML file to pass to an extract or plot command. For example:
```
audio:
  min_freq: 350
  max_freq: 4000
  win_length: .08
  spec_height: 192
  spec_width: 256
```
Note that the window length is specified as a fraction of a second, so .08 seconds in this example. That way the real window length does not vary if you change the sampling rate. As a rule of thumb, the sampling rate should be about 2.1 times the maximum frequency. Before training your first model, it is advisable to examine some spectrogram images and choose settings that seem reasonable as a starting point. For example, the frequency range needed for your application may be greater or less than the defaults.

The SpecGroup table allows you to easily experiment with different spectrogram settings. Running `extract-all` or `extract-by-image` creates spectrograms assigned to the default SpecGroup, if none is specified. Once you have curated some training data, use the `reextract` command to create another set of spectrograms, assigned to a different SpecGroup. That way you can keep spectrograms with different settings for easy experimentation.
## Training
The `pickle` command creates a binary pickle file (`data/training.pkl` by default), which is the source of training data for the `train` command. Reading a binary file is much faster than querying the database, so this speeds up the training process. Also, this provides a simple way to select a SpecGroup, and/or a subset of classes for training. For training, you should always provide a config file to override some defaults. Here is an expanded version of the earlier example:
```
train:
  train_pickle: "data/low_freq.pkl"
  model_type: "effnet.4"
  head_type: "basic_sed"
  learning_rate: .002
  drop_rate: 0.1
  drop_path_rate: 0.1
  val_portion: 0.1
  num_epochs: 20
```
The model_type parameter can be "timm.x" for any model x supported by [timm](https://github.com/huggingface/pytorch-image-models). However, many bioacoustic recognizers benefit from a smaller model than typical timm models. Therefore BriteKit provides a set of scalable models, such as "effnet.3" and "effnet.4", where larger numbers indicate larger models. The scalable models are:
| Model | Original Name | Comments | Original Paper |
|---|---|---|---|
| dla | DLA | Slow and not good for large models, but often a good choice for very small models. | [here](https://arxiv.org/abs/1707.06484) |
| effnet | EfficientNetV2 | Medium speed, widely used, useful for all sizes. | [here](https://arxiv.org/abs/2104.00298) |
| gernet | GerNet | Fast, useful for all but the smallest models. | [here](https://arxiv.org/abs/2006.14090) |
| hgnet |  HgNetV2| Fast, useful for all but the smallest models. | not published |
| vovnet | VovNet  | Medium-fast, useful for all sizes. | [here](https://arxiv.org/abs/1904.09730) |

For very small models, say with less than 10 classes and just a few thousand training spectrograms, DLA and VovNet are good candidates. As model size increases, DLA becomes slower and less appropriate.

If `head_type` is not specified, BriteKit uses the default classifier head defined by the model. However, you can also specify any of the following head types:
| Head Type | Description |
|---|---|
| basic | A basic non-SED classifier head. |
| effnet | The classifier head used in EfficientNetV2. |
| hgnet | The classifier head used in HgNetV2. |
| basic_sed | A basic SED head. |
| scalable_sed | The basic_sed head can be larger than desired.  |

Specifying head_type="effnet" is sometimes helpful for other models such as DLA and VovNet. See the discussion of [Backbones and Classifier Heads](#backbones-and-classifier-heads) below for more information.

You can specify val_portion > 0 to run validation on a portion of the training data, or num_folds > 1 to run k-fold cross-validation. In the latter case, training output will be in logs/fold-0, logs/fold-1 etc. Otherwise output is under logs/fold-0. Output from the first training run is saved in logs/fold-0/version_0, and the version number is incremented in subsequent runs. To view graphs of the loss and learning rate, type `tensorboard --logdir <log directory>`. This will launch an embedded web server and display a URL that you can use to access it from a web browser.

## Testing
To run a test, you need to annotate a set of test recordings, analyze them with your model or ensemble, and then run the `rpt-test` command. Annotations must be saved in a CSV file with a defined format. We recommend annotating each relevant sound (per-segment), but you can also do per-minute and per-recording annotations to save time. Per-recording annotations are defined in a CSV file with these columns:
| Column | Description |
|---|---|
| recording | Just the stem of the recording name, e.g. XC12345, not XC12345.mp3. |
| classes | Defined classes found in the recording, separated by commas. For example: AMCR,BCCH,COYE.

Per-minute annotations are defined in a CSV file with these columns:
| Column | Description |
|---|---|
| recording | Just the stem of the recording name, as above. |
| minute | 1 for the first minute, 2 for the second, etc. |
| classes | Defined classes found in that minute, separated by commas.

Per-segment annotations are recommended, and are defined in a CSV file with these columns:
| Column | Description |
|---|---|
| recording | Just the stem of the recording name, as above. |
| class | Identified class.
| start_time | Where the sound starts, in seconds from the start of the recording.
| end_time | Where the sound ends, in seconds from the start of the recording.

Use the `analyze` command to analyze the recordings with your model or ensemble. For testing, be sure to specify `--min_score 0`. That way all predictions will be saved, not just those above a particular threshold, which is important when calculating metrics. See [Metrics (PR-AUC and ROC-AUC)](#metrics-pr-auc-and-roc-auc) for more information.

It's usually best for a test to consist of a single directory of recordings, containing a file called annotations.csv. If that directory is called recordings and you run analyze specifying `--output recordings/labels`, you could generate test reports as follows:
```
britekit rpt-test -a recordings/annotations.csv -l labels -o <output-dir>
```
If your annotations were per-minute or per-recording, you would specify the `--granularity minute` or `--granularity recording` argument (`--granularity segment` is the default).
## Tuning
Before tuning your model, you need to create a good test, as described in the previous section. Then you can use the `tune` command to find optimal settings for a given test. If you are only tuning inference parameters, you can run many iterations very quickly, since no training is needed. To tune training hyperparameters, many training runs are needed, which takes longer. You can also use the `tune` command to tune audio/spectrogram settings. In that case, every iteration extracts a new set of spectrograms, which takes even longer.

Here is a practical approach:
1. Review spectrogram plots with different settings, especially spec_duration, spec_width, spec_height, min_frequency, max_frequency and win_length. Then choose reasonable-looking initial settings. For example, if all the relevant sounds fall between 1000 and 5000 Hz, set min and max frequency accordingly.
2. Do an initial tuning pass of the main training hyperparameters, especially model_type, head_type and num_epochs.
3. Based on the above, carefully tune the audio/spectrogram parameters.

This usually leads to a substantial improvement in scores (see [Metrics (PR-AUC and ROC-AUC)](#metrics-pr-auc-and-roc-auc), and then you can proceed to fine-tuning the training and inference. For inference, it is usually worth tuning the `audio_power` parameter. If you are using a SED classifier head, it is also worth tuning `segment_len` and `overlap`. For training, it may be worth tuning the data augmentation hyperparameters, which are described in detail in the [Data Augmentation](#data-augmentation) section below.

To run the `tune` command, you would typically use a config YAML file as described earlier, plus a special tuning YAML file, as in this example:
```
- name: spec_width
  type: int
  bounds:
  - 256
  - 512
  step: 64
```
This gives the name of the parameter to tune, its data type, and the bounds and step sizes to try. In this case, we want to try spec_width values of 256, 320, 384, 448 and 512. You can also tune multiple parameters at the same time, by simply appending more definitions similar to this one. Parameters that have a choice of defined values rather than a range are specified like this:
```
- name: head_type
  type: categorical
  choices:
  - "effnet"
  - "hgnet"
  - "basic_sed"
```
When running the `tune` command, you can ask it to test all defined combinations based on the input, or to test a random sample. To try 100 random combinations, add the argument `--tries 100`. To tune audio/spectrogram parameters, add the `--extract` argument. To tune inference only, add the `--notrain` argument.

Training is non-deterministic, and results for a given group of settings can vary substantially across multiple training runs. Therefore it is important to specify the `--runs` argument, indicating how often training should be run for a given set of values.

As an example, to find the best `spec_width` value, we could type a command like this:
```
britekit tune -c yaml/my_train.yml -p yaml/my_tune.yml -a my_test/annotations.csv -o output/tune-spec-width --runs 5 --extract
```
This will perform an extract before each trial, and use the average score from 5 training runs in each case. Scores will be based on the given test, using macro-averaged ROC-AUC, although this can be changed with the `--metric` argument.

## Ensembling
TBD

## Calibrating
TBD

# More Information

-----

## Spectrograms
TBD
## Backbones and Classifier Heads
TBD
## Metrics (PR-AUC and ROC-AUC)
TBD
## Data Augmentation
TBD
## Development Environment
TBD