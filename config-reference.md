## Configuration Reference
### Sections
- [AudioConfig](#audioconfig)
- [TrainingConfig](#trainingconfig)
- [InferenceConfig](#inferenceconfig)
- [MiscConfig](#miscconfig)
- [BaseConfig](#baseconfig)

### AudioConfig
| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `spec_duration` | `float` | 5.0 | Spectrogram duration in seconds |
| `spec_height` | `int` | 128 | Spectrogram height in pixels |
| `spec_width` | `int` | 480 | Spectrogram width in pixels (must be divisible by 32) |
| `win_length` | `float` | 0.055 | Window length is specified in seconds, to retain temporal and frequency resolution when max_freq and sampling rate are changed |
| `n_fft` | `Union[int, NoneType]` | None | If none, set n_fft = win_length_samples |
| `max_freq` | `int` | 8000 | Maximum frequency in Hz |
| `min_freq` | `int` | 100 | Minimum frequency in Hz |
| `sampling_rate` | `int` | 18000 | A little more than 2 * max_freq |
| `freq_scale` | `str` | 'mel' | "linear", "log" or "mel" |
| `power` | `float` | 1.0 | Use 1.0 for magnitude and 2.0 for power spectrograms |
| `decibels` | `bool` | False | Use decibel amplitude scale? |
| `top_db` | `float` | 80 | Threshold below max amplitude in dB; lower values are clipped |
| `db_power` | `float` | 1.0 | Raise to this exponent after convert to decibels |
| `log_freq_gain` | `float` | 0.6 | Boost loudness of higher frequencies with log scale |
| `mel_norm` | `Union[str, NoneType]` | None | Mel filterbank normalization: None or "slaney" |
| `choose_channel` | `bool` | False | Use heuristic to pick the cleanest audio channel? |
| `check_seconds` | `float` | 6.0 | Check this many seconds to pick channel |
| `use_spec_cache` | `bool` | False | When use_spec_cache=False, each spectrogram is generated separately. When use_spec_cache=True, a single spectrogram is generated for the recording, by concatenating chunks. Then that big spectrogram is divided as needed, which saves time when there is a lot of overlap. |
| `chunks_per_spec` | `int` | 3 | chunks_per_spec defines the size of the individual spectrograms that are concatenated to create the cache, where the duration is chunks_per_spec * spec_duration. |

### TrainingConfig
| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `model_type` | `str` | 'effnet.2' | Use timm.x for timm model "x" |
| `head_type` | `Union[str, NoneType]` | None | If None, use backbone's default |
| `hidden_channels` | `int` | 256 | Hidden channels in classifier head |
| `lse_temp` | `float` | 0.5 | LSE temperature for temporal_sed head |
| `two_way` | `bool` | True | Bi-directional or unidirectional temporal_sed head |
| `pretrained` | `bool` | False | Use pretrained weights (applies to timm models) |
| `load_ckpt_path` | `Union[str, NoneType]` | None | For transfer learning or fine-tuning |
| `freeze_backbone` | `bool` | False | Option when transfer learning |
| `multi_label` | `bool` | True | True for multi-label, False for multi-class classification |
| `deterministic` | `bool` | False | Enable deterministic training for reproducibility |
| `seed` | `Union[int, NoneType]` | None | Random seed for reproducibility; None uses random seed |
| `learning_rate` | `float` | 0.001 | Base learning rate |
| `batch_size` | `int` | 64 | Mini-batch size |
| `shuffle` | `bool` | True | Shuffle data during training? |
| `num_epochs` | `int` | 10 | Number of epochs |
| `warmup_fraction` | `float` | 0.0 | Learning rate warmup fraction |
| `save_last_n` | `int` | 3 | Save checkpoints for this many last epochs |
| `num_folds` | `int` | 1 | For k-fold cross-validation |
| `val_portion` | `float` | 0 | Used only if num_folds = 1 |
| `train_db` | `str` | 'data/training.db' | Path to training database |
| `train_pickle` | `str` | 'data/training.pkl' | Path to training pickle file |
| `test_pickle` | `Union[str, NoneType]` | None | Path to test pickle file |
| `frame_label_pickle` | `Union[str, NoneType]` | None | Path to frame-label pickle used in SED training |
| `num_workers` | `int` | 3 | Number of trainer worker threads |
| `compile` | `bool` | False | Compile the model? |
| `mixed_precision` | `bool` | False | Use mixed precision? |
| `use_class_weights` | `bool` | False | Should loss function weight classes by spec count? |
| `weight_exponent` | `float` | 0.5 | Exponent to soften the class weights |
| `pos_label_smoothing` | `float` | 0.08 | Positive side of asymmetric label smoothing |
| `neg_label_smoothing` | `float` | 0.01 | Negative side of asymmetric label smoothing |
| `optimizer` | `str` | 'radam' | Any timm optimizer |
| `opt_weight_decay` | `Union[float, NoneType]` | None | Weight decay option (L2 regularization) |
| `opt_beta1` | `Union[float, NoneType]` | None | Adam/RAdam beta1 (exponential decay rate for first moment) |
| `opt_beta2` | `Union[float, NoneType]` | None | Adam/RAdam beta2 (exponential decay rate for second moment) |
| `drop_rate` | `Union[float, NoneType]` | None | Standard dropout |
| `drop_path_rate` | `Union[float, NoneType]` | None | Stochastic depth dropout |
| `sed_fps` | `int` | 4 | Frames per second from SED heads |
| `frame_loss_weight` | `float` | 0.5 | Segment_loss_weight = 1 - frame_loss_weight |
| `offpeak_weight` | `float` | 0 | Weight for penalizing predictions outside peak regions |
| `max_per_recording` | `Union[int, NoneType]` | None | Per-recording sampling: if set, randomly select this many specs per recording per epoch |
| `val_max_per_recording` | `Union[int, NoneType]` | None | Per-recording limit for validation: if set, take the first N specs per recording |
| `augment` | `bool` | True | Use data augmentation? |
| `max_augmentations` | `int` | 1 | Up to this many per spectrogram |
| `noise_class_name` | `str` | 'Noise' | Augmentation treats noise specially |
| `prob_simple_merge` | `float` | 0.32 | Prob of simple merge |
| `prob_mixup` | `float` | 0.0 | Prob of traditional mixup (mutually exclusive with simple merge) |
| `prob_cutmix` | `float` | 0.0 | Prob of CutMix (mutually exclusive with simple merge and mixup) |
| `mixup_alpha` | `float` | 0.4 | Beta distribution parameter for mixup/cutmix lambda |
| `prob_fade1` | `float` | 0.5 | Prob of fading after augmentation |
| `min_fade1` | `float` | 0.1 | Min factor for fading |
| `max_fade1` | `float` | 1.0 | Max factor for fading |
| `augmentations` | `list` | <factory <lambda>> | Detailed augmentation settings |

### InferenceConfig
| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `segment_len` | `Union[float, NoneType]` | None | For models with SED heads, if segment_len is None, output tags of variable lengths that match the sounds detected, otherwise output tags of length segment_len seconds. For non-SED models, segment_len is defined by the model. |
| `overlap` | `float` | 0.0 | Number of seconds overlap for adjacent spectrograms |
| `min_score` | `float` | 0.8 | Only generate labels when score is at least this |
| `num_threads` | `int` | 3 | More threads = faster but more VRAM |
| `max_models` | `Union[int, NoneType]` | None | If specified, limit ensemble size accordingly |
| `autocast` | `bool` | True | Faster and less VRAM but less precision |
| `audio_power` | `float` | 0.7 | Audio power parameter during inference |
| `scaling_coefficient` | `float` | 1.0 | Platt scaling coefficient, to align predictions with probabilities |
| `scaling_intercept` | `float` | 0.0 | Platt scaling intercept, to align predictions with probabilities |
| `label_field` | `str` | 'codes' | "names", "codes", "alt_names" or "alt_codes" |
| `block_size` | `int` | 200 | Do this many spectrograms at a time to avoid running out of GPU memory |
| `openvino_block_size` | `int` | 100 | Block size when OpenVINO is used (do not change after creating onnx files) |
| `initial_offsets` | `Union[list, NoneType]` | None | If specified, analyze command calls get_overlapping_scores instead of get_recording_scores and passes this array. |

### MiscConfig
| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `force_cpu` | `bool` | False | If true, use CPU (for performance comparisons) |
| `ckpt_folder` | `str` | 'data/ckpt' | Use an ensemble of all checkpoints in this folder for inference |
| `search_ckpt_path` | `Union[str, NoneType]` | None | Folder with one or more checkpoints for embeddings and search |
| `exclude_list` | `str` | 'data/exclude.txt' | Classes listed in this file are excluded from inference output |
| `source_regexes` | `Union[list, NoneType]` | <factory <lambda>> | Sample regexes to map recording names to source names |
| `map_codes` | `Union[dict, NoneType]` | None | Dict mapping old to new class codes for checkpoint compatibility |

### BaseConfig
| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `audio` | `AudioConfig` | <factory AudioConfig> |  |
| `train` | `TrainingConfig` | <factory TrainingConfig> |  |
| `infer` | `InferenceConfig` | <factory InferenceConfig> |  |
| `misc` | `MiscConfig` | <factory MiscConfig> |  |
