from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Audio:
    spec_duration: float = 5.0  # Spectrogram duration in seconds
    spec_height: int = 128  # Spectrogram height
    spec_width: int = 480  # Spectrogram width (divisible by 32)

    # Window length is specified in seconds,
    # to retain temporal and frequency resolution
    # when max_freq and sampling rate are changed
    win_length: float = 0.055

    max_freq: int = 8000  # Maximum frequency for spectrograms
    min_freq: int = 100  # Minimum frequency for spectrograms
    sampling_rate: int = 18000  # A little more than 2 * max_freq
    choose_channel: bool = True  # Use heuristic to pick the cleanest audio channel
    check_seconds: float = 3.0  # Use this many when picking cleanest channel
    freq_scale: str = "mel"  # "linear", "log" or "mel"
    power: float = 1.0  # Use 1.0 for magnitude and 2.0 for power spectrograms
    decibels: bool = False  # Use decibel amplitude scale?
    top_db: float = 80  # Parameter to decibel conversion
    db_power: float = 1.0  # Raise to this exponent after convert to decibels
    log_freq_gain: float = 0.6  # Boost loudness of higher frequencies with log scale


@dataclass
class Training:
    # model selection parameters
    model_type: str = "effnet.2"  # Use timm.x for timm model "x"
    head_type: Optional[str] = None  # If None, use backbone's default
    hidden_channels: int = 256  # Used by some non-default classifier heads
    pretrained: bool = False  # For group=timm
    load_ckpt_path: Optional[str] = None  # For transfer learning or fine-tuning
    freeze_backbone: bool = False  # Option when transfer learning

    # general training parameters
    multi_label: bool = True  # Multi-label or multi-class?
    deterministic: bool = False  # Deterministic training?
    seed: Optional[int] = None  # Training seed
    learning_rate: float = 0.001  # Base learning rate
    batch_size: int = 64  # Mini-batch size
    shuffle: bool = True  # Shuffle data during training?
    num_epochs: int = 10  # Number of epochs
    warmup_fraction: float = 0.0  # Learning rate warmup fraction
    save_last_n: int = 3  # Save checkpoints for this many last epochs
    num_folds: int = 1  # For k-fold cross-validation
    val_portion: float = 0  # Used only if num_folds = 1
    train_db: str = "data/training.db"  # Path to training database
    train_pickle: str = "data/training.pkl"  # Path to training pickle file
    test_pickle: Optional[str] = None  # Path to test pickle file
    num_workers: int = 3  # Number of trainer worker threads
    compile: bool = False  # Compile the model?
    mixed_precision: bool = False  # Use mixed precision?

    pos_label_smoothing: float = 0.08  # Positive side of asymmetric label smoothing
    neg_label_smoothing: float = 0.01  # Negative side of asymmetric label smoothing

    # optimizer parameters; other good choices are
    # "adam" with decay = 1e-6
    # "adamp" with decay = 0
    optimizer: str = "radam"  # Any timm optimizer
    opt_weight_decay: float = 1e-6  # Weight decay option (L2 regularization)
    opt_beta1: float = 0.9  # Optimizer parameter
    opt_beta2: float = 0.999  # Optimizer parameter

    # dropout parameters are passed to model only if not None
    drop_rate: Optional[float] = None  # Standard dropout
    drop_path_rate: Optional[float] = None  # Stochastic depth dropout

    # SED-specific parameters
    sed_fps: int = 4  # Frames per second from SED heads
    frame_loss_weight: float = 0.5  # Segment_loss_weight = 1 - frame_loss_weight

    # data augmentation
    augment: bool = True  # Use data augmentation?
    noise_class_name: str = "Noise"  # Augmentation treats noise specially
    prob_simple_merge: float = 0.32  # Prob of simple merge
    prob_fade1: float = 0.5  # Prob of fading after augmentation
    min_fade1: float = 0.1  # Min factor for fading
    max_fade1: float = 1.0  # Max factor for fading

    # Loss penalty weight for SED models
    offpeak_weight: float = 0.002

    # Detailed augmentation settings
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
    # Number of seconds overlap for adjacent spectrograms
    overlap: float = 0.0
    min_score: float = 0.80  # Only generate labels when score is at least this
    num_threads: int = 3  # More threads = faster but more VRAM
    autocast: bool = True  # Faster and less VRAM but less precision
    audio_power: float = 0.7  # Audio power parameter during inference
    # Platt scaling coefficient, to align predictions with probabilities
    scaling_coefficient: float = 1.0
    # Platt scaling intercept, to align predictions with probabilities
    scaling_intercept: float = 0.0
    label_field: str = "codes"  # "names", "codes", "alt_names" or "alt_codes"
    # Do this many spectrograms at a time to avoid running out of GPU memory
    block_size: int = 200
    # Block size when OpenVINO is used (do not change after creating onnx files)
    openvino_block_size: int = 100
    seed: int = 99  # Reduce non-determinism during inference

    # These parameters control a second pass during inference.
    # If lower_min_if_confirmed is true, count the number of seconds for a class in a recording,
    # where score >= min_score + raise_min_to_confirm * (1 - min_score).
    # If seconds >= confirmed_if_seconds, the class is assumed to be present, so scan again,
    # lowering the min_score by multiplying it by lower_min_factor.
    lower_min_if_confirmed: bool = True
    # To be confirmed, score must be >= min_score + this * (1 - min_score)
    raise_min_to_confirm: float = 0.5
    # Need at least this many confirmed seconds >= raised threshold
    confirmed_if_seconds: float = 8.0
    # If so, include all labels with score >= this * min_score
    lower_min_factor: float = 0.6


@dataclass
class Miscellaneous:
    force_cpu: bool = False  # If true, use CPU (for performance comparisons)
    # Use an ensemble of all checkpoints in this folder for inference
    ckpt_folder: str = "data/ckpt"
    # Checkpoint used in searching and clustering
    search_ckpt_path: Optional[str] = None
    # List of classes used to generate pickle files
    classes_file: str = "data/classes.txt"
    # Classes listed in this file are ignored in analysis
    ignore_file: str = "data/ignore.txt"

    # Sample regexes to map recording names to source names
    source_regexes: Optional[list] = field(
        default_factory=lambda: [
            ("^[A-Za-z0-9_-]{11}-\\d+$", "Audioset"),
            ("^XC\\d+$", "Xeno-Canto"),
            ("^N\\d+$", "iNaturalist"),
            ("^\\d+$", "Macaulay Library"),
            (".*", "default"),
        ]
    )

    map_names: Optional[dict] = None  # Map old class names to new names
    map_codes: Optional[dict] = None  # Map old class codes to new codes


@dataclass
class BaseConfig:
    audio: Audio = field(default_factory=Audio)
    train: Training = field(default_factory=Training)
    infer: Inference = field(default_factory=Inference)
    misc: Miscellaneous = field(default_factory=Miscellaneous)


@dataclass
# TODO: allow API users to replace some functions with their own here.
# Callables cannot be included in BaseConfig, since they are not serializable
class FunctionConfig:
    not_defined_yet: Optional[Callable] = None
