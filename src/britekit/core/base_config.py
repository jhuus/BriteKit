#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AudioConfig:
    spec_duration: float = 5.0  # Spectrogram duration in seconds
    spec_height: int = 128  # Spectrogram height in pixels
    spec_width: int = 480  # Spectrogram width in pixels (must be divisible by 32)

    # Window length is specified in seconds,
    # to retain temporal and frequency resolution
    # when max_freq and sampling rate are changed
    win_length: float = 0.055
    n_fft: Optional[int] = None  # If none, set n_fft = win_length_samples

    max_freq: int = 8000  # Maximum frequency in Hz
    min_freq: int = 100  # Minimum frequency in Hz
    sampling_rate: int = 18000  # A little more than 2 * max_freq
    freq_scale: str = "mel"  # "linear", "log" or "mel"
    power: float = 1.0  # Use 1.0 for magnitude and 2.0 for power spectrograms
    decibels: bool = False  # Use decibel amplitude scale?
    top_db: float = 80  # Threshold below max amplitude in dB; lower values are clipped
    db_power: float = 1.0  # Raise to this exponent after convert to decibels
    log_freq_gain: float = 0.6  # Boost loudness of higher frequencies with log scale

    mel_norm: Optional[str] = None  # Mel filterbank normalization: None or "slaney"

    choose_channel: bool = False  # Use heuristic to pick the cleanest audio channel?
    check_seconds: float = 6.0  # Check this many seconds to pick channel

    # When use_spec_cache=False, each spectrogram is generated separately.
    # When use_spec_cache=True, a single spectrogram is generated for the recording,
    # by concatenating chunks. Then that big spectrogram is divided as needed,
    # which saves time when there is a lot of overlap.
    use_spec_cache: bool = False

    # chunks_per_spec defines the size of the individual spectrograms that
    # are concatenated to create the cache, where the duration is
    # chunks_per_spec * spec_duration.
    chunks_per_spec: int = 3


@dataclass
class TrainingConfig:
    # Model selection parameters
    model_type: str = "effnet.2"  # Use timm.x for timm model "x"
    head_type: Optional[str] = None  # If None, use backbone's default
    hidden_channels: int = 256  # Used by some non-default classifier heads
    pretrained: bool = False  # Use pretrained weights (applies to timm models)
    load_ckpt_path: Optional[str] = None  # For transfer learning or fine-tuning
    freeze_backbone: bool = False  # Option when transfer learning

    # General training parameters

    # True for multi-label, False for multi-class classification
    multi_label: bool = True
    deterministic: bool = False  # Enable deterministic training for reproducibility
    seed: Optional[int] = None  # Random seed for reproducibility; None uses random seed
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

    # Path to frame-label pickle used in SED training
    frame_label_pickle: Optional[str] = None

    num_workers: int = 3  # Number of trainer worker threads
    compile: bool = False  # Compile the model?
    mixed_precision: bool = False  # Use mixed precision?

    # Should loss function weight classes by spec count?
    use_class_weights: bool = False
    weight_exponent: float = 0.5  # Exponent to soften the class weights

    pos_label_smoothing: float = 0.08  # Positive side of asymmetric label smoothing
    neg_label_smoothing: float = 0.01  # Negative side of asymmetric label smoothing

    # Optimizer parameters
    optimizer: str = "radam"  # Any timm optimizer
    opt_weight_decay: Optional[float] = None  # Weight decay option (L2 regularization)

    # Adam/RAdam beta1 (exponential decay rate for first moment)
    opt_beta1: Optional[float] = None

    # Adam/RAdam beta2 (exponential decay rate for second moment)
    opt_beta2: Optional[float] = None

    # Dropout parameters are passed to model only if not None
    drop_rate: Optional[float] = None  # Standard dropout
    drop_path_rate: Optional[float] = None  # Stochastic depth dropout

    # SED-specific parameters
    sed_fps: int = 4  # Frames per second from SED heads
    frame_loss_weight: float = 0.5  # Segment_loss_weight = 1 - frame_loss_weight

    # Weight for penalizing predictions outside peak regions
    offpeak_weight: float = 0.002

    # Epsilon threshold for absence penalty calculation
    absence_penalty_eps: float = 0.2
    absence_penalty_tau: float = 7.0  # Temperature scaling factor for absence penalty
    absence_penalty_weight: float = 0.0  # Absence penalty weight for SED models

    # Per-recording sampling: if set, randomly select this many specs per recording per epoch
    max_per_recording: Optional[int] = None

    # Per-recording limit for validation: if set, take the first N specs per recording
    val_max_per_recording: Optional[int] = None

    # Data augmentation
    augment: bool = True  # Use data augmentation?
    max_augmentations: int = 1  # Up to this many per spectrogram
    noise_class_name: str = "Noise"  # Augmentation treats noise specially
    prob_simple_merge: float = 0.32  # Prob of simple merge
    # Prob of traditional mixup (mutually exclusive with simple merge)
    prob_mixup: float = 0.0
    # Prob of CutMix (mutually exclusive with simple merge and mixup)
    prob_cutmix: float = 0.0
    mixup_alpha: float = 0.4  # Beta distribution parameter for mixup/cutmix lambda
    prob_fade1: float = 0.5  # Prob of fading after augmentation
    min_fade1: float = 0.1  # Min factor for fading
    max_fade1: float = 1.0  # Max factor for fading

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
                "params": {"min_std": 0.01, "max_std": 0.1},
            },
            {
                "name": "blur",
                "prob": 0,
                "params": {"min_sigma": 0.5, "max_sigma": 1.0},
            },
            {
                "name": "flip_horizontal",
                "prob": 0,
                "params": {},
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
class InferenceConfig:
    # For models with SED heads, if segment_len is None, output tags of variable lengths
    # that match the sounds detected, otherwise output tags of length segment_len seconds.
    # For non-SED models, segment_len is defined by the model.
    segment_len: Optional[float] = None
    # Number of seconds overlap for adjacent spectrograms
    overlap: float = 0.0
    min_score: float = 0.80  # Only generate labels when score is at least this
    num_threads: int = 3  # More threads = faster but more VRAM
    max_models: Optional[int] = None  # If specified, limit ensemble size accordingly
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

    # If specified, analyze command calls get_overlapping_scores instead of
    # get_recording_scores and passes this array.
    initial_offsets: Optional[list] = None


@dataclass
class MiscConfig:
    force_cpu: bool = False  # If true, use CPU (for performance comparisons)
    # Use an ensemble of all checkpoints in this folder for inference
    ckpt_folder: str = "data/ckpt"
    # Folder with one or more checkpoints for embeddings and search
    search_ckpt_path: Optional[str] = None
    # Classes listed in this file are excluded from inference output
    exclude_list: str = "data/exclude.txt"

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

    # Dict mapping old to new class codes for checkpoint compatibility
    map_codes: Optional[dict] = None


@dataclass
class BaseConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    infer: InferenceConfig = field(default_factory=InferenceConfig)
    misc: MiscConfig = field(default_factory=MiscConfig)
