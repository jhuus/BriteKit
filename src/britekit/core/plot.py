# Defer some imports to improve initialization performance.
from typing import Optional

from britekit.core.config_loader import get_config


def plot_spec(
    spec,
    output_path: str,
    show_dims: bool = False,
    spec_duration: Optional[float] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
):
    """
    Plot and save a spectrogram image.

    Args:
    - spec (np.ndarray): Spectrogram of shape (height, width)
    - output_path (str): Path to save the image (e.g., "output.png")
    - show_dims (bool): Whether to show frequency and time scales
    - spec_duration (float, optional): Number of seconds represented.
    - height (int, optional): Output image height in pixels. If not specified,
        the existing square behavior is preserved.
    - width (int, optional): Output image width in pixels. If not specified,
        the existing square behavior is preserved.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    cfg = get_config()
    assert cfg.audio.freq_scale in {"linear", "log", "mel"}

    if spec_duration is None:
        spec_duration = cfg.audio.spec_duration

    if spec.ndim == 3:
        spec = spec.reshape((cfg.audio.spec_height, cfg.audio.spec_width))

    if show_dims:
        x_incr = 64
        spec_width = spec.shape[1]
        x_tick_locations = [i for i in range(0, spec_width + 1, x_incr)]
        x_tick_labels = [
            f"{i/(spec_width / spec_duration):.1f}s"
            for i in range(0, spec_width + 1, x_incr)
        ]

        # generate a y_tick for first and last frequencies and every n kHz
        if cfg.audio.freq_scale == "linear":
            freq_incr = (cfg.audio.max_freq - cfg.audio.min_freq) / (
                cfg.audio.spec_height - 1
            )
            frequencies = np.arange(
                cfg.audio.min_freq, cfg.audio.max_freq + freq_incr, freq_incr
            )
        elif cfg.audio.freq_scale == "log":
            log2_f_min = np.log2(cfg.audio.min_freq)
            log2_f_max = np.log2(cfg.audio.max_freq)
            log2_ticks = np.linspace(log2_f_min, log2_f_max, cfg.audio.spec_height)
            frequencies = 2**log2_ticks
        else:
            # mel scale
            import librosa

            frequencies = librosa.mel_frequencies(
                n_mels=cfg.audio.spec_height,
                fmin=cfg.audio.min_freq,
                fmax=cfg.audio.max_freq,
            )

        # simple heuristic for placement of y-axis frequency labels
        freq_diff = cfg.audio.max_freq - cfg.audio.min_freq
        if freq_diff >= 4000:
            units = 1000
        elif freq_diff >= 2000:
            units = 200
        elif freq_diff >= 500:
            units = 100
        elif freq_diff >= 100:
            units = 20
        else:
            units = 10

        y_tick_locations = []
        y_tick_labels = []
        mult = 1
        for i, freq in enumerate(frequencies):
            if i == 0:
                y_tick_locations.append(i)
                y_tick_labels.append(f"{int(freq)} Hz")
            elif i == len(frequencies) - 1:
                y_tick_locations.append(i)
                if units == 1000:
                    y_tick_labels.append(f"{int(round(freq, 0) / units)} kHz")
                else:
                    y_tick_labels.append(f"{int(freq)} Hz")
            elif freq >= mult * units:
                if units == 1000:
                    y_tick_labels.append(f"{mult} kHz")
                else:
                    y_tick_labels.append(f"{int(freq)} Hz")
                if abs(freq - mult * units) < abs(frequencies[i - 1] - mult * units):
                    y_tick_locations.append(i)
                else:
                    y_tick_locations.append(i - 1)

                mult += 1

    # Optionally set explicit output dimensions (in pixels).
    # Defaults preserve prior behavior (square output without explicit sizing).
    DPI = 100
    plt.clf()  # clear any existing plot data
    if height is not None and width is not None:
        fig = plt.gcf()
        fig.set_dpi(DPI)
        fig.set_size_inches(width / DPI, height / DPI)

    # 'flat' is much faster than 'gouraud'
    plt.pcolormesh(spec, shading="flat")
    if show_dims:
        plt.xticks(x_tick_locations, x_tick_labels)
        plt.yticks(y_tick_locations, y_tick_labels)
        if height is not None and width is not None:
            # Preserve requested size; avoid tight bbox which can change dimensions
            plt.savefig(output_path)
        else:
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    else:
        plt.tick_params(
            axis="both",  # apply to both axes
            which="both",  # apply to both major and minor ticks
            bottom=False,  # disable ticks
            top=False,
            left=False,
            right=False,
            labelbottom=False,  # disable labels
            labeltop=False,
            labelleft=False,
            labelright=False,
        )
        if height is not None and width is not None:
            # Preserve requested size; avoid tight bbox which can change dimensions
            plt.savefig(output_path)
        else:
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)

    plt.close()
