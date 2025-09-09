# BriteKit

-----

## Table of Contents

- [License](#license)
- [Installation](#installation)
- [Introduction](#introduction)
- [Configuration](#configuration)
- [Downloading Recordings](#downloading-recordings)
- [Managing Training Data](#managing-training-data)

## Reference documentation:
- [Command Reference](command-reference.md)
- [Command API Reference](command-api-reference.md)
- [General API Reference](api-reference.md)
- [Configuration Reference](config-reference.md)

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
- testing, tuning and calibration
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

