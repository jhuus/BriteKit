from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Audio:
    spec_duration: float = 5.0  # spectrogram duration in seconds
    spec_height: int = 128  # spectrogram height
    spec_width: int = 480  # spectrogram width (divisible by 5 and 32)

    # window length is specified in seconds,
    # to retain temporal and frequency resolution
    # when max_freq and sampling rate are changed
    win_length: float = 0.055

    # max_freq should be a multiple of spec_width / spec_duration
    # for perfect temporal alignment
    max_freq: int = 8000  # maximum frequency for spectrograms
    min_freq: int = 100  # minimum frequency for spectrograms
    sampling_rate: int = 18000  # a little more than 2 * max_freq
    choose_channel: bool = True  # use heuristic to pick the cleanest audio channel
    check_seconds: float = 3.0  # use this many when picking cleanest channel
    freq_scale: str = "mel"  # "linear", "log" or "mel"
    power: float = 1.0
    decibels: bool = False  # use decibel amplitude scale?
    top_db: float = 80  # parameter to decibel conversion
    db_power: float = 1.0  # raise to this exponent after convert to decibels


@dataclass
class Training:
    # model selection parameters
    model_type: str = "effnet.2"  # use timm.x for timm model "x"
    head_type: Optional[str] = None  # if None, use backbone's default
    hidden_channels: int = 256  # used by some non-default classifier heads
    pretrained: bool = False  # for group=timm
    load_ckpt_path: Optional[str] = None  # for transfer learning or fine-tuning
    freeze_backbone: bool = False

    # general training parameters
    multi_label: bool = True
    deterministic: bool = False
    seed: Optional[int] = None
    learning_rate: float = 0.001  # base learning rate (see the "find-lr" command)
    batch_size: int = 64
    shuffle: bool = True
    num_epochs: int = 10
    warmup_fraction: float = 0.0
    save_last_n: int = 3  # save checkpoints for this many last epochs
    num_folds: int = 1  # for k-fold cross-validation
    val_portion: float = 0  # used only if num_folds = 1
    train_db: str = "data/training.db"  # path to training database
    train_pickle: Optional[str] = None
    test_pickle: Optional[str] = None
    num_workers: int = 2
    compile: bool = False
    mixed_precision: bool = False

    # asymmetric label smoothing (separate positive and negative)
    pos_label_smoothing: float = 0.08
    neg_label_smoothing: float = 0.01

    # optimizer parameters; other good choices are
    # "adam" with decay = 1e-6
    # "adamp" with decay = 0
    optimizer: str = "radam"  # any timm optimizer
    opt_weight_decay: float = 1e-6
    opt_beta1: float = 0.9
    opt_beta2: float = 0.999

    # dropout parameters are passed to model only if not None
    drop_rate: Optional[float] = None  # standard dropout
    drop_path_rate: Optional[float] = None  # stochastic depth dropout

    # SED-specific parameters
    sed_fps: int = 4  # frames per second from SED heads
    frame_loss_weight: float = 0.5  # segment_loss_weight = 1 - frame_loss_weight

    # data augmentation
    augment: bool = True
    noise_class_name: str = "Noise"
    prob_simple_merge: float = 0.32
    prob_fade1: float = 0.5  # prob of fading after augmentation
    min_fade1: float = 0.1
    max_fade1: float = 1.0

    # loss penalty weight for SED models
    offpeak_weight: float = 0.002

    augmentations: list = field(
        default_factory=lambda: [
            {
                "name": "add_real_noise",
                "prob": 0.34,
                "params": {"prob_fade2": 0.5, "min_fade2": 0.2, "max_fade2": 0.8},
            },
            {
                "name": "add_white_noise",
                "prob": 0,
                "params": {"std1": 0.08},
            },
            {
                "name": "flip_horizontal",
                "prob": 0,
                "params": {},
            },
            {
                "name": "freq_blur",
                "prob": 0,
                "params": {"sigma": 0.4},
            },
            {
                "name": "freq_mask",
                "prob": 0,
                "params": {"max_width1": 4},
            },
            {
                "name": "shift_horizontal",
                "prob": 0.6,
                "params": {"max_shift": 8},
            },
            {
                "name": "speckle",
                "prob": 0,
                "params": {"std2": 0.1},
            },
            {
                "name": "time_mask",
                "prob": 0,
                "params": {"max_width2": 8},
            },
        ]
    )


@dataclass
class Inference:
    # For models with SED heads, if segment_len is None, output tags of variable lengths
    # that match the sounds detected, otherwise output tags of length segment_len seconds.
    # For non-SED models, segment_len is defined by the model.
    segment_len: Optional[float] = None
    # number of seconds overlap for adjacent spectrograms
    overlap: float = 0.0
    min_score: float = 0.80  # only generate labels when score is at least this
    num_threads: int = 3  # more threads = faster but more VRAM
    autocast: bool = True  # faster and less VRAM but less precision
    audio_power: float = 0.7  # audio power parameter during inference
    # Platt scaling coefficient, to align predictions with probabilities
    scaling_coefficient: float = 1.0
    # Platt scaling intercept, to align predictions with probabilities
    scaling_intercept: float = 0.0
    label_field: str = "codes"  # "names", "codes", "alt_names" or "alt_codes"
    # do this many spectrograms at a time to avoid running out of GPU memory
    block_size: int = 200
    # block size when OpenVINO is used (do not change after creating onnx files)
    openvino_block_size: int = 100
    seed: int = 99  # reduce non-determinism during inference

    # These parameters control a second pass during inference.
    # If lower_min_if_confirmed is true, count the number of seconds for a class in a recording,
    # where score >= min_score + raise_min_to_confirm * (1 - min_score).
    # If seconds >= confirmed_if_seconds, the class is assumed to be present, so scan again,
    # lowering the min_score by multiplying it by lower_min_factor.
    lower_min_if_confirmed: bool = True
    # to be confirmed, score must be >= min_score + this * (1 - min_score)
    raise_min_to_confirm: float = 0.5
    # need at least this many confirmed seconds >= raised threshold
    confirmed_if_seconds: float = 8.0
    # if so, include all labels with score >= this * min_score
    lower_min_factor: float = 0.6


@dataclass
class Miscellaneous:
    force_cpu: bool = False  # if true, use CPU (for performance comparisons)
    # use an ensemble of all checkpoints in this folder for inference
    ckpt_folder: str = "data/ckpt"
    # checkpoint used in searching and clustering
    search_ckpt_path: str = "data/ckpt-search"
    # list of classes used to generate pickle files
    classes_file: str = "data/classes.txt"
    # classes listed in this file are ignored in analysis
    ignore_file: str = "data/ignore.txt"
    source_regexes: Optional[list] = None
    # sample regexes to map recording names to source names
    source_regexes = field(
        default_factory=lambda: [
            ("^[A-Za-z0-9_-]{11}-\\d+$", "Audioset"),
            ("^XC\\d+$", "Xeno-Canto"),
            ("^N\\d+$", "iNaturalist"),
            ("^\\d+$", "Macaulay Library"),
            (".*", "default"),
        ]
    )

    # map old class names and codes to new names and codes
    map_names: Optional[dict] = None
    map_codes: Optional[dict] = None


@dataclass
class BaseConfig:
    audio: Audio = field(default_factory=Audio)
    train: Training = field(default_factory=Training)
    infer: Inference = field(default_factory=Inference)
    misc: Miscellaneous = field(default_factory=Miscellaneous)


@dataclass
# Callables cannot be included in BaseConfig, since they are not serializable
class FunctionConfig:
    # print, log, echo, ...
    echo: Optional[Callable] = None
