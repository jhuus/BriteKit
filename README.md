# BriteKit

-----

## Quick Guide

- [License](#license)
- [Installation](#installation)
- [Introduction](#introduction)
- [Configuration](#configuration)
- [Downloading Recordings](#downloading-recordings)
- [Managing Training Data](#managing-training-data)
- [Training](#training)
- [Testing](#testing)
- [Tuning](#tuning)
- [Ensembling](#ensembling)
- [Calibrating](#calibrating)

## In-Depth Topics:
- [Spectrograms](#spectrograms)
- [Backbones and Classifier Heads](#backbones-and-classifier-heads)
- [Metrics (PR-AUC and ROC-AUC)](#metrics-pr-auc-and-roc-auc)

## Reference Documentation:

- [Command Reference](command-reference.md)
- [Command API Reference](command-api-reference.md)
- [General API Reference](api-reference.md)
- [Configuration Reference](config-reference.md)

# Quick Guide

-----

## License
BriteKit is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
## Installation
Install the BriteKit package using pip:
```console
pip install britekit
```
Once BriteKit is installed, initialize a working environment using the `init` command:
```console
britekit init --dest=<directory path>
```
This creates the directories needed and installs sample files. If you omit `--dest`, it will create
directories under the current working directory.
## Introduction
BriteKit (Bioacoustic Recognizer Technology Kit) is a Python package that facilitates the development of bioacoustic recognizers using deep learning.
It provides a command-line interface (CLI) as well as a Python API, to support functions such as:
- downloading recordings from Xeno-Canto, iNaturalist, and Youtube (optionally with Google Audioset metadata)
- managing training data in a SQLite database
- training models
- testing, tuning and calibrating your models
- reporting
- deployment

To view a list of BriteKit commands, type `britekit --help`. You can also get help for individual commands, e.g. `britekit train --help` describes the `train` command.
When accessing BriteKit from Python, the `britekit.commands` namespace contains a function for each command, as documented [here](command-api-reference.md).
The classes used by the commands can also be accessed, and are documented [here](api-reference.md).
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
This overrides the default values for model_type, learning_rate, drop_rate and num_epochs.
## Downloading Recordings
The `inat`, `xeno` and `youtube` commands make it easy to download recordings from Xeno_Canto, iNaturalist and Youtube. For iNaturalist it is important to provide the scientific name. For example, to download recordings of the American Green Frog (lithobates clamitans), type:
```
britekit inat --name "lithobates clamitans" --output <output-path>
```
For Xeno-Canto, use `--name` for the common name or `--sci` for the scientific name. For Youtube, specify the ID of the corresponding video. For example, specify `--id K_EsxukdNXM` to download the audio from https://www.youtube.com/watch?v=K_EsxukdNXM.

BriteKit also supports downloads using [Google Audioset](https://research.google.com/audioset/), which is metadata that classifies sounds in Youtube videos. Audioset was released in March 2017, so any videos uploaded later than that are not included. Also, some videos that are tagged in Audioset are no longer available. Type `britekit audioset --help` for more information.
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

For very small models, say with less than 10 classes and just a few thousand training spectrograms, DLA and VovNet are good candidates. As model size increases, DLA becomes slower and less accurate.

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
TBD

## Tuning
TBD

## Ensembling
TBD

## Calibrating
TBD

# In-Depth Topics

-----

## Spectrograms
TBD
## Backbones and Classifier Heads
TBD
## Metrics (PR-AUC and ROC-AUC)
TBD
