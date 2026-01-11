#!/usr/bin/env python3

# Defer some imports to improve initialization performance.
import logging
from typing import Any, Optional

from britekit.core.base_config import BaseConfig
from britekit.core.config_loader import get_config
from britekit.core.util import get_device

# suppress some torchaudio warnings
import warnings

warnings.filterwarnings("ignore", message=".*TorchCodec.*|.*StreamingMediaDecoder.*")


def load_audio(path, sr):
    """Fast audio load for general use."""

    import torchaudio as ta  # faster than librosa
    import numpy as np

    try:
        # Load (channels, samples), float32
        signal, sr0 = ta.load(path)

        # Resample if needed
        if sr0 != sr:
            signal = ta.functional.resample(signal, sr0, sr)

        # Make it mono
        signal = signal.numpy()
        if signal.ndim == 2:
            signal = signal.mean(axis=0)

        # Ensure 1D float32
        signal = np.asarray(signal, dtype=np.float32)

    except Exception as e:
        logging.error(f"Caught exception in audio load of {path}: {e}")
        return None

    return signal


class Audio:
    """
    Provide methods for reading audio recordings and creating spectrograms.

    Attributes:
        device (str, optional): Device supported by pytorch ('cuda', 'cpu' or 'mps')
        cfg (optional): BaseConfig to use.
    """

    def __init__(self, device: Optional[str] = None, cfg: Optional[BaseConfig] = None):
        if device is None:
            self.device = get_device()
        else:
            self.device = device

        self.linear_transform_cache: dict[tuple, Any] = {}
        self.mel_transform_cache: dict[tuple, Any] = {}
        self.log2_filterbank_cache: dict[tuple, Any] = {}
        self.cached = None
        self.path = None
        self.signal: Any = None
        self.set_config(cfg)
        self.sampling_rate = self.cfg.audio.sampling_rate  # in case resampling needed

    def set_config(self, cfg: Optional[BaseConfig] = None, resample: bool = True):
        """
        Set or update the audio configuration for spectrogram generation.

        This method configures the audio processing parameters including sampling rate,
        window length, frequency scales, and transforms. It should be called before
        generating spectrograms to ensure proper configuration.

        The main bottleneck in audio processing is the load function. We often want to
        generate spectrograms of different forms from the same audio, so it's best to
        only load it once. This can be accomplished by calling set_config to update
        spectrogram config before calling get_spectrograms with new settings.

        If max_frequency is changing, it is best to start with the highest frequency,
        so we downsample rather than upsampling.

        Args:
        - cfg (Optional[BaseConfig]): Configuration object. If None, uses default config.
        - resample (bool): If true and sampling rate has changed, resample the audio.
        """
        import torch
        import torchaudio as ta

        if cfg is None:
            self.cfg = get_config()
        else:
            self.cfg = cfg

        # convert window length from seconds to frames;
        # defining it in seconds retains temporal and frequency
        # resolution when sampling rate changes
        self.win_length = int(self.cfg.audio.win_length * self.cfg.audio.sampling_rate)

        self.cached = None  # can not trust that cache is suitable
        if (
            resample
            and self.signal is not None
            and self.sampling_rate != self.cfg.audio.sampling_rate
        ):
            logging.debug(
                "Audio::set_config resample from %d to %d",
                self.sampling_rate,
                self.cfg.audio.sampling_rate,
            )

            signal = ta.functional.resample(
                torch.from_numpy(self.signal),
                self.sampling_rate,
                self.cfg.audio.sampling_rate,
            )
            self.signal = signal.cpu().numpy()

        self.sampling_rate = self.cfg.audio.sampling_rate

        # force power scale when converting to decibels
        if self.cfg.audio.decibels:
            self.cfg.audio.power = 2

        logging.debug(
            "Audio::set_config sr=%d, win=%d, duration=%.2f, height=%d, width=%d",
            self.sampling_rate,
            self.win_length,
            self.cfg.audio.spec_duration,
            self.cfg.audio.spec_height,
            self.cfg.audio.spec_width,
        )

        # cache transforms and filterbanks for performance
        key = (
            self.sampling_rate,
            self.win_length,
            self.cfg.audio.spec_duration,
            self.cfg.audio.spec_height,
            self.cfg.audio.spec_width,
        )

        if key in self.linear_transform_cache:
            self.linear_transform = self.linear_transform_cache[key]
        else:
            self.linear_transform = ta.transforms.Spectrogram(
                n_fft=2 * self.win_length,
                win_length=self.win_length,
                hop_length=int(
                    self.cfg.audio.spec_duration
                    * self.sampling_rate
                    / self.cfg.audio.spec_width
                ),
                power=self.cfg.audio.power,
            ).to(self.device)
            self.linear_transform_cache[key] = self.linear_transform

        if key in self.log2_filterbank_cache:
            self.log2_filterbank = self.log2_filterbank_cache[key]
        else:
            self.log2_filterbank = self._make_log2_filterbank()
            self.log2_filterbank_cache[key] = self.log2_filterbank

        # previous transforms are needed in choose_channel, but mel may not be needed
        if self.cfg.audio.freq_scale == "mel":
            if key in self.mel_transform_cache:
                self.mel_transform = self.mel_transform_cache[key]
            else:
                self.mel_transform = ta.transforms.MelSpectrogram(
                    sample_rate=self.sampling_rate,
                    n_fft=2 * self.win_length,
                    win_length=self.win_length,
                    hop_length=int(
                        self.cfg.audio.spec_duration
                        * self.sampling_rate
                        / self.cfg.audio.spec_width
                    ),
                    f_min=self.cfg.audio.min_freq,
                    f_max=self.cfg.audio.max_freq,
                    n_mels=self.cfg.audio.spec_height,
                    power=self.cfg.audio.power,
                ).to(self.device)
                self.mel_transform_cache[key] = self.mel_transform

    def load(self, path):
        """
        Load audio from the given recording file.

        Loads an audio file and stores it in the Audio object for subsequent
        spectrogram generation. Supports both mono and stereo recordings.
        For stereo recordings, can automatically choose the cleaner channel
        if choose_channel is enabled in the configuration.

        Args:
        - path (str): Path to the audio recording file.

        Returns:
            tuple: (signal, sampling_rate) where:
                - signal: The loaded audio signal as numpy array
                - sampling_rate: The sampling rate (should equal cfg.audio.sampling_rate)

        Note:
            If loading fails, signal will be None and an error will be logged.
        """
        import torchaudio as ta  # faster than librosa
        import numpy as np

        if not path or not isinstance(path, str):
            logging.error(f"Invalid path provided: {path}")
            return None, self.cfg.audio.sampling_rate

        if path == self.path:
            # already loaded this recording
            logging.debug("Audio::load skip load of %s", path)
            return self.signal, self.cfg.audio.sampling_rate

        try:
            # Load (channels, samples), float32
            logging.debug("Audio::load processing %s", path)
            self.path = path
            waveform, sr = ta.load(path)

            # Resample if needed
            if sr != self.cfg.audio.sampling_rate:
                waveform = ta.functional.resample(
                    waveform, sr, self.cfg.audio.sampling_rate
                )

            # Handle channels
            waveform = waveform.numpy()
            if waveform.ndim == 2 and waveform.shape[0] == 2:
                # stereo recording
                if self.cfg.audio.choose_channel:
                    self.cached = None  # important to set this before choose-channel
                    waveform = self._choose_channel(waveform[0], waveform[1])
                else:
                    waveform = waveform.mean(axis=0)

            # Ensure 1D float32
            self.signal = np.asarray(waveform, dtype=np.float32)
            if self.signal.ndim == 2:
                self.signal = self.signal.squeeze(0)  # (1, len) -> (len,)

        except Exception as e:
            self.signal = None
            self.path = None
            logging.error(f"Caught exception in audio load of {path}: {e}")

        self.cached = None  # important to set this after choose-channel too
        return self.signal, self.sampling_rate

    def get_spectrograms(
        self,
        start_times: list[float],
        spec_duration: Optional[float] = None,
        freq_scale: Optional[str] = None,
        decibels: Optional[float] = None,
        top_db: Optional[int] = None,
        db_power: Optional[int] = None,
    ):
        """
        Generate normalized and unnormalized spectrograms for specified time offsets.

        Creates spectrograms from the loaded audio signal at the specified start times.
        Supports different frequency scales (linear, log, mel) and optional decibel conversion.
        Returns both normalized (0-1 range) and unnormalized versions of the spectrograms.

        Args:
        - start_times (list[float]): List of start times in seconds from the beginning
            of the recording for each spectrogram.
        - spec_duration (Optional[float]): Length of each spectrogram in seconds.
            Defaults to cfg.audio.spec_duration.
        - freq_scale (Optional[str]): Frequency scale to use ('linear', 'log', 'mel').
            Defaults to cfg.audio.freq_scale.
        - decibels (Optional[float]): Whether to convert to decibels.
            Defaults to cfg.audio.decibels.
        - top_db (Optional[int]): Maximum decibel value for normalization.
            Defaults to cfg.audio.top_db.
        - db_power (Optional[int]): Power to apply after decibel conversion.
            Defaults to cfg.audio.db_power.

        Returns:
            tuple: (normalized_specs, unnormalized_specs) where:
                - normalized_specs: Spectrograms normalized to 0-1 range (torch.Tensor)
                - unnormalized_specs: Original spectrograms without normalization (torch.Tensor)

        Note:
            Returns (None, None) if no audio signal is loaded.
        """
        import numpy as np

        if self.signal is None or start_times is None:
            return None, None

        if spec_duration is not None and spec_duration <= 0:
            return None, None

        if spec_duration is None:
            # this is not the same as spec_duration=self.cfg.audio.spec_duration in the parameter list,
            # since self.cfg.audio.spec_duration can be modified after the parameter list is evaluated
            spec_duration = self.cfg.audio.spec_duration

        # Get one spectrogram for the whole recording, then split it up
        if self.cached is None:
            logging.debug("Audio::get_spectrograms generate spectrogram for recording")
            self.cached = self._get_raw_spectrogram(
                self.signal,
                freq_scale=freq_scale,
                decibels=decibels,
                top_db=top_db,
                db_power=db_power,
            )
        else:
            logging.debug("Audio::get_spectrograms reuse cached spectrogram")

        assert self.cached is not None
        samples_per_sec = self.sampling_rate / self.linear_transform.hop_length
        specs = []
        for i, offset in enumerate(start_times):
            start_sample = int(offset * samples_per_sec)
            end_sample = int((offset + spec_duration) * samples_per_sec)
            end_sample = min(end_sample, self.cached.shape[1])
            if end_sample - start_sample < samples_per_sec:
                break  # require at least one second of audio in a spectrogram

            if start_sample < self.cached.shape[1]:
                spec = self.cached[:, start_sample:end_sample]
                if spec.shape[1] > self.cfg.audio.spec_width:
                    spec = spec[:, : self.cfg.audio.spec_width]
                elif spec.shape[1] < self.cfg.audio.spec_width:
                    pad_width = self.cfg.audio.spec_width - spec.shape[1]
                    spec = np.pad(spec, ((0, 0), (0, pad_width)), mode="constant")

                specs.append(spec)

        # Handle empty specs list to prevent error
        if not specs:
            return np.empty(0), np.empty(0)

        unnormalized_specs = np.stack(specs, axis=0)
        normalized_specs = unnormalized_specs.copy()
        self._normalize(normalized_specs)
        return normalized_specs, unnormalized_specs

    def seconds(self):
        """
        Get the duration of the loaded audio signal in seconds.

        Returns:
            float: Duration of the signal in seconds, or 0 if no signal is loaded.
        """
        if self.signal is None or self.cfg.audio.sampling_rate == 0:
            return 0.0

        return self.signal_len() / self.cfg.audio.sampling_rate

    def signal_len(self):
        """
        Get the length of the loaded audio signal in samples.

        Returns:
            int: Number of samples in the signal, or 0 if no signal is loaded.
        """
        return 0 if self.signal is None else len(self.signal)

    # =============================================================================
    # Private Helper Methods
    # =============================================================================

    def _make_log2_filterbank(self, bins_per_octave=12):
        import numpy as np
        import torch

        f_min = max(self.cfg.audio.min_freq, 1.0)  # prevent log2(0) exception
        f_max = self.cfg.audio.max_freq
        n_bins = self.cfg.audio.spec_height
        sr = self.cfg.audio.sampling_rate
        n_fft = 2 * self.win_length

        fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
        fft_log2 = np.log2(fft_freqs + 1e-6)  # avoid log(0)

        log2_fmin = np.log2(f_min)
        log2_fmax = np.log2(f_max)
        log2_centers = np.linspace(log2_fmin, log2_fmax, n_bins)

        sigma_log2 = 1.0 / bins_per_octave  # 1 bin = 1/bins_per_octave octave

        filters = []
        for log_cf in log2_centers:
            # Gaussian filter in log2(f)
            weight = np.exp(-0.5 * ((fft_log2 - log_cf) / sigma_log2) ** 2)

            # Normalize per filter (contrast-preserving)
            weight /= np.sum(weight) + 1e-12

            # Boost power in higher frequencies, so more like mel scale;
            # increase log_freq_gain to increase the boost
            cf_hz = 2**log_cf
            gain = (cf_hz / f_min) ** self.cfg.audio.log_freq_gain
            weight *= gain

            filters.append(weight)

        filters = np.array(filters)
        return torch.tensor(filters, dtype=torch.float32).to(self.device)

    def _get_raw_spectrogram(
        self,
        signal,
        freq_scale: Optional[str] = None,
        decibels: Optional[float] = None,
        top_db: Optional[float] = None,
        db_power: Optional[float] = None,
    ):
        import torch
        import torchaudio as ta
        import numpy as np

        if freq_scale is None:
            freq_scale = self.cfg.audio.freq_scale

        assert freq_scale in {"linear", "log", "mel"}

        if decibels is None:
            decibels = self.cfg.audio.decibels

        if top_db is None:
            top_db = self.cfg.audio.top_db

        if db_power is None:
            db_power = self.cfg.audio.db_power

        # Process in chunks to avoid CUDA memory errors on long recordings.
        # 5 minutes at 32kHz = 9.6M samples.
        max_chunk_samples = 5 * 60 * self.cfg.audio.sampling_rate

        if signal.shape[0] <= max_chunk_samples:
            # Short signal: process all at once
            spec = self._compute_spectrogram_chunk(signal, freq_scale)
        else:
            # Long signal: process in chunks and concatenate
            chunks = []
            for start in range(0, signal.shape[0], max_chunk_samples):
                end = min(start + max_chunk_samples, signal.shape[0])
                chunk_signal = signal[start:end]
                if chunk_signal.shape[0] < max_chunk_samples:
                    chunk_signal = np.pad(
                        chunk_signal, (0, max_chunk_samples - chunk_signal.shape[0])
                    )

                chunk_spec = self._compute_spectrogram_chunk(chunk_signal, freq_scale)
                chunks.append(chunk_spec)
                del chunk_spec
                torch.cuda.empty_cache()

            spec = np.concatenate(chunks, axis=1)

        if decibels:
            # Apply dB conversion on CPU (already numpy)
            spec = torch.from_numpy(spec).unsqueeze(0)  # [1, F, T]
            spec = ta.transforms.AmplitudeToDB(stype="power", top_db=top_db)(spec)
            spec = spec**db_power
            spec = spec[0].numpy()

        return spec

    def _compute_spectrogram_chunk(self, signal, freq_scale):
        """Compute spectrogram for a single chunk of audio, returning numpy array."""
        import torch
        import torch.nn.functional as F

        signal = signal.reshape((1, signal.shape[0]))
        tensor = torch.from_numpy(signal).to(self.device)

        if freq_scale == "log":
            spec = self.linear_transform(tensor)  # [1, n_freqs, n_frames]
            spec = torch.matmul(
                self.log2_filterbank, spec.squeeze(0)
            )  # [n_mels, n_frames]
            spec = spec.unsqueeze(0)  # [1, n_mels, n_frames]

        elif freq_scale == "mel":
            spec = self.mel_transform(tensor)  # [1, n_mels, T]

        elif freq_scale == "linear":
            spec = self.linear_transform(tensor)
            freqs = torch.fft.rfftfreq(
                2 * self.win_length, d=1 / self.cfg.audio.sampling_rate
            )
            mask = (freqs >= self.cfg.audio.min_freq) & (
                freqs <= self.cfg.audio.max_freq
            )
            spec = spec[:, mask, :].unsqueeze(1)  # [1, 1, F_sel, T]

            # downsample frequency to spec_height (energy-preserving)
            spec = F.interpolate(
                spec,
                size=(self.cfg.audio.spec_height, spec.shape[-1]),
                mode="area",
            )
            spec = spec.squeeze(1)  # [1, F, T]

        result = spec[0].cpu().numpy()
        del spec, tensor
        return result

    def _normalize(self, specs):
        """Normalize values between 0 and 1."""
        for i in range(len(specs)):
            if specs[i] is None:
                continue

            max = specs[i].max()
            if max > 0:
                specs[i] = specs[i] / max

    def _choose_channel(self, left_signal, right_signal):
        """
        Stereo recordings sometimes have one clean channel and one noisy one;
        so rather than just merge them, use heuristics to pick the cleaner one.
        This heuristic was developed by training an sklearn DecisionTreeClassifier,
        converting the tree to if/else statements, changing it to treat the left
        and right channels symmetrically, and finally retuning the hyperparameters.
        """
        from . import audio_util
        import numpy as np

        # if one channel is null, use the other
        left_sum = left_signal.sum()
        right_sum = right_signal.sum()
        if left_sum == 0 and right_sum != 0:
            return right_signal
        elif left_sum != 0 and right_sum == 0:
            return left_signal

        recording_seconds = int(len(left_signal) / self.cfg.audio.sampling_rate)
        check_seconds = min(recording_seconds, self.cfg.audio.check_seconds)
        if check_seconds == 0:
            return left_signal  # make an arbitrary choice

        self.signal = left_signal
        left_specs, _ = self.get_spectrograms(
            [0],
            spec_duration=check_seconds,
            freq_scale="log",
            decibels=True,
            top_db=80,
            db_power=1,
        )
        left_spec = left_specs[0]
        self.signal = right_signal
        right_specs, _ = self.get_spectrograms(
            [0],
            spec_duration=check_seconds,
            freq_scale="log",
            decibels=True,
            top_db=80,
            db_power=1,
        )
        right_spec = right_specs[0]

        # calculate sum and median per channel
        left_sum = left_spec.sum()
        right_sum = right_spec.sum()

        left_median = np.median(left_spec)
        right_median = np.median(right_spec)

        # calculate energy in defined band per channel
        cfg = self.cfg.audio
        band_min_freq = min(cfg.max_freq, max(cfg.min_freq, cfg.energy_min_freq))
        band_max_freq = max(cfg.min_freq, min(cfg.max_freq, cfg.energy_max_freq))
        left_energy = audio_util.band_limited_energy(
            left_spec,
            sr=cfg.sampling_rate,
            freq_range=(band_min_freq, band_max_freq),
            f_min=cfg.min_freq,
            f_max=cfg.max_freq,
        )
        right_energy = audio_util.band_limited_energy(
            right_spec,
            sr=cfg.sampling_rate,
            freq_range=(band_min_freq, band_max_freq),
            f_min=cfg.min_freq,
            f_max=cfg.max_freq,
        )

        # set parameters based on the low-energy channel
        if left_energy <= right_energy:
            low_energy, high_energy = left_signal, right_signal
            sum_ratio = 1 if right_sum == 0 else left_sum / right_sum
            median_ratio = 1 if right_median == 0 else left_median / right_median
        else:
            low_energy, high_energy = right_signal, left_signal
            sum_ratio = 1 if left_sum == 0 else right_sum / left_sum
            median_ratio = 1 if left_median == 0 else right_median / left_median

        # apply the heuristic
        if median_ratio <= cfg.median_threshold:
            if sum_ratio <= 1 / cfg.sum_threshold:
                return high_energy
            else:
                return low_energy
        else:
            if sum_ratio <= cfg.sum_threshold:
                return low_energy
            else:
                return high_energy
