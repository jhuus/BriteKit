## Configuration Reference
### Sections
- [Audio](#audio)
- [Training](#training)
- [Inference](#inference)
- [Miscellaneous](#miscellaneous)
- [BaseConfig](#baseconfig)
- [FunctionConfig](#functionconfig)

### Audio
| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `spec_duration` | `float` | 3.0 | spectrogram duration in seconds |
| `spec_height` | `int` | 192 | spectrogram height |
| `spec_width` | `int` | 384 | spectrogram width (3 * 128) |
| `win_length` | `float` | 0.055 | window length is specified in seconds, to retain temporal and frequency resolution when max_freq and sampling rate are changed |
| `max_freq` | `int` | 13000 | maximum frequency for spectrograms |
| `min_freq` | `int` | 200 | minimum frequency for spectrograms |
| `sampling_rate` | `int` | 28000 | a little more than 2 * max_freq |
| `choose_channel` | `bool` | True | use heuristic to pick the cleanest audio channel |
| `check_seconds` | `float` | 3.0 | use this many when picking cleanest channel |
| `freq_scale` | `str` | 'mel' | "linear", "log" or "mel" |
| `power` | `float` | 1.0 |  |
| `decibels` | `bool` | False | use decibel amplitude scale? |
| `top_db` | `float` | 80 | parameter to decibel conversion |
| `db_power` | `float` | 1.0 | raise to this exponent after convert to decibels |

### Training
| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `model_type` | `str` | 'effnet.5' | use timm.x for timm model "x" |
| `head_type` | `Union[str, NoneType]` | None | if None, use backbone's default |
| `hidden_channels` | `int` | 256 | used by some non-default classifier heads |
| `pretrained` | `bool` | False | for group=timm |
| `load_ckpt_path` | `Union[str, NoneType]` | None | for transfer learning or fine-tuning |
| `freeze_backbone` | `bool` | False |  |
| `multi_label` | `bool` | True | general training parameters |
| `deterministic` | `bool` | False |  |
| `seed` | `Union[int, NoneType]` | None |  |
| `learning_rate` | `float` | 0.001 | base learning rate (see the "find-lr" command) |
| `batch_size` | `int` | 64 |  |
| `shuffle` | `bool` | True |  |
| `num_epochs` | `int` | 10 |  |
| `warmup_fraction` | `float` | 0.0 |  |
| `save_last_n` | `int` | 3 | save checkpoints for this many last epochs |
| `num_folds` | `int` | 1 | for k-fold cross-validation |
| `val_portion` | `float` | 0 | used only if num_folds = 1 |
| `train_db` | `str` | 'data/training.db' | path to training database |
| `train_pickle` | `Union[str, NoneType]` | None |  |
| `test_pickle` | `Union[str, NoneType]` | None |  |
| `num_workers` | `int` | 2 |  |
| `compile` | `bool` | False |  |
| `mixed_precision` | `bool` | False |  |
| `pos_label_smoothing` | `float` | 0.08 | asymmetric label smoothing (separate positive and negative) |
| `neg_label_smoothing` | `float` | 0.01 |  |
| `optimizer` | `str` | 'radam' | any timm optimizer |
| `opt_weight_decay` | `float` | 1e-06 |  |
| `opt_beta1` | `float` | 0.9 |  |
| `opt_beta2` | `float` | 0.999 |  |
| `drop_rate` | `Union[float, NoneType]` | None | standard dropout |
| `drop_path_rate` | `Union[float, NoneType]` | None | stochastic depth dropout |
| `sed_fps` | `int` | 4 | frames per second from SED heads |
| `frame_loss_weight` | `float` | 0.5 | segment_loss_weight = 1 - frame_loss_weight |
| `augment` | `bool` | True | data augmentation |
| `noise_class_name` | `str` | 'Noise' |  |
| `prob_simple_merge` | `float` | 0.32 |  |
| `prob_fade1` | `float` | 0.5 | prob of fading after augmentation |
| `min_fade1` | `float` | 0.1 |  |
| `max_fade1` | `float` | 1.0 |  |
| `offpeak_weight` | `float` | 0.002 | loss penalty weight for SED models |
| `augmentations` | `list` | <factory <lambda>> |  |

### Inference
| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `segment_len` | `Union[float, NoneType]` | None | For models with SED heads, if segment_len is None, output tags of variable lengths that match the sounds detected, otherwise output tags of length segment_len seconds. For non-SED models, segment_len is defined by the model. |
| `num_threads` | `int` | 3 | more threads = faster but more VRAM |
| `autocast` | `bool` | True | faster and less VRAM but less precision |
| `audio_power` | `float` | 0.7 | audio power parameter during inference |
| `spec_overlap_seconds` | `float` | 0.0 | number of seconds overlap for adjacent spectrograms |
| `min_score` | `float` | 0.8 | only generate labels when score is at least this |
| `scaling_coefficient` | `float` | 1.0 | Platt scaling coefficient, to align predictions with probabilities |
| `scaling_intercept` | `float` | 0.0 | Platt scaling intercept, to align predictions with probabilities |
| `label_field` | `str` | 'codes' | "names", "codes", "alt_names" or "alt_codes" |
| `block_size` | `int` | 200 | do this many spectrograms at a time to avoid running out of GPU memory |
| `openvino_block_size` | `int` | 100 | block size when OpenVINO is used (do not change after creating onnx files) |
| `seed` | `int` | 99 | reduce non-determinism during inference |
| `lower_min_if_confirmed` | `bool` | True | These parameters control a second pass during inference. If lower_min_if_confirmed is true, count the number of seconds for a species in a recording, where score >= min_score + raise_min_to_confirm * (1 - min_score). If seconds >= confirmed_if_seconds, the species is assumed to be present, so scan again, lowering the min_score by multiplying it by lower_min_factor. |
| `raise_min_to_confirm` | `float` | 0.5 | to be confirmed, score must be >= min_score + this * (1 - min_score) |
| `confirmed_if_seconds` | `float` | 8.0 | need at least this many confirmed seconds >= raised threshold |
| `lower_min_factor` | `float` | 0.6 | if so, include all labels with score >= this * min_score |

### Miscellaneous
| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `force_cpu` | `bool` | False | if true, use CPU (for performance comparisons) |
| `ckpt_folder` | `str` | 'data/ckpt' | use an ensemble of all checkpoints in this folder for inference |
| `search_ckpt_path` | `str` | 'data/ckpt-search/v0-e9.ckpt' | checkpoint used in searching and clustering |
| `classes_file` | `str` | 'data/classes.txt' | list of classes used to generate pickle files |
| `ignore_file` | `str` | 'data/ignore.txt' | classes listed in this file are ignored in analysis |
| `source_regexes` | `Union[list, NoneType]` | <factory <lambda>> |  |
| `map_names` | `Union[dict, NoneType]` | None | map old species names and codes to new names and codes |
| `map_codes` | `Union[dict, NoneType]` | None |  |

### BaseConfig
| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `audio` | `Audio` | <factory Audio> |  |
| `train` | `Training` | <factory Training> |  |
| `infer` | `Inference` | <factory Inference> |  |
| `misc` | `Miscellaneous` | <factory Miscellaneous> |  |

### FunctionConfig
| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `echo` | `Union[Callable[], NoneType]` | None | print, log, echo, ... |
