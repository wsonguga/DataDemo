# -*- coding: utf-8 -*-
import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..scg import scg_peaks
from ..epochs import epochs_to_df
from ..signal import signal_fixpeaks
from ..stats import rescale
from .scg_segment import scg_segment


def scg_plot(scg_signals, rpeaks=None, sampling_rate=None, show_type="default"):
    """Visualize scg data.

    Parameters
    ----------
    scg_signals : DataFrame
        DataFrame obtained from `scg_process()`.
    rpeaks : dict
        The samples at which the R-peak occur. Dict returned by
        `scg_process()`. Defaults to None.
    sampling_rate : int
        The sampling frequency of the scg (in Hz, i.e., samples/second). Needs to be supplied if the
        data should be plotted over time in seconds. Otherwise the data is plotted over samples.
        Defaults to None. Must be specified to plot artifacts.
    show_type : str
        Visualize the scg data with 'default' or visualize artifacts thresholds with 'artifacts' produced by
        `scg_fixpeaks()`, or 'full' to visualize both.

    Returns
    -------
    fig
        Figure representing a plot of the processed scg signals (and peak artifacts).

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> scg = nk.scg_simulate(duration=15, sampling_rate=1000, heart_rate=80)
    >>> signals, info = nk.scg_process(scg, sampling_rate=1000)
    >>> nk.scg_plot(signals, sampling_rate=1000, show_type='default') #doctest: +ELLIPSIS
    <Figure ...>

    See Also
    --------
    scg_process

    """
    # Sanity-check input.
    if not isinstance(scg_signals, pd.DataFrame):
        raise ValueError(
            "NeuroKit error: scg_plot(): The `scg_signals` argument must be the "
            "DataFrame returned by `scg_process()`."
        )

    # Extract R-peaks.
    peaks = np.where(scg_signals["scg_R_Peaks"] == 1)[0]

    # Prepare figure and set axes.
    if show_type in ["default", "full"]:
        if sampling_rate is not None:
            x_axis = np.linspace(0, scg_signals.shape[0] / sampling_rate, scg_signals.shape[0])
            gs = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[1 - 1 / np.pi, 1 / np.pi])
            fig = plt.figure(constrained_layout=False)
            ax0 = fig.add_subplot(gs[0, :-1])
            ax1 = fig.add_subplot(gs[1, :-1])
            ax2 = fig.add_subplot(gs[:, -1])
            ax0.set_xlabel("Time (seconds)")
            ax1.set_xlabel("Time (seconds)")
            ax2.set_xlabel("Time (seconds)")
        else:
            x_axis = np.arange(0, scg_signals.shape[0])
            fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
            ax0.set_xlabel("Samples")
            ax1.set_xlabel("Samples")

        fig.suptitle("Electrocardiogram (scg)", fontweight="bold")
        plt.subplots_adjust(hspace=0.3, wspace=0.1)

        # Plot cleaned, raw scg, R-peaks and signal quality.
        ax0.set_title("Raw and Cleaned Signal")

        quality = rescale(
            scg_signals["scg_Quality"], to=[np.min(scg_signals["scg_Clean"]), np.max(scg_signals["scg_Clean"])]
        )
        minimum_line = np.full(len(x_axis), quality.min())

        # Plot quality area first
        ax0.fill_between(
            x_axis, minimum_line, quality, alpha=0.12, zorder=0, interpolate=True, facecolor="#4CAF50", label="Quality"
        )

        # Plot signals
        ax0.plot(x_axis, scg_signals["scg_Raw"], color="#B0BEC5", label="Raw", zorder=1)
        ax0.plot(x_axis, scg_signals["scg_Clean"], color="#E91E63", label="Cleaned", zorder=1, linewidth=1.5)
        ax0.scatter(x_axis[peaks], scg_signals["scg_Clean"][peaks], color="#FFC107", label="R-peaks", zorder=2)

        # Optimize legend
        handles, labels = ax0.get_legend_handles_labels()
        order = [2, 0, 1, 3]
        ax0.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="upper right")

        # Plot heart rate.
        ax1.set_title("Heart Rate")
        ax1.set_ylabel("Beats per minute (bpm)")

        ax1.plot(x_axis, scg_signals["scg_Rate"], color="#FF5722", label="Rate", linewidth=1.5)
        rate_mean = scg_signals["scg_Rate"].mean()
        ax1.axhline(y=rate_mean, label="Mean", linestyle="--", color="#FF9800")

        ax1.legend(loc="upper right")

        # Plot individual heart beats.
        if sampling_rate is not None:
            ax2.set_title("Individual Heart Beats")

            heartbeats = scg_segment(scg_signals["scg_Clean"], peaks, sampling_rate)
            heartbeats = epochs_to_df(heartbeats)

            heartbeats_pivoted = heartbeats.pivot(index="Time", columns="Label", values="Signal")

            ax2.plot(heartbeats_pivoted)

            cmap = iter(
                plt.cm.YlOrRd(np.linspace(0, 1, num=int(heartbeats["Label"].nunique())))  # pylint: disable=E1101
            )  # Aesthetics of heart beats

            lines = []
            for x, color in zip(heartbeats_pivoted, cmap):
                (line,) = ax2.plot(heartbeats_pivoted[x], color=color)
                lines.append(line)

    # Plot artifacts
    if show_type in ["artifacts", "full"]:
        if sampling_rate is None:
            raise ValueError(
                "NeuroKit error: scg_plot(): Sampling rate must be specified for artifacts" " to be plotted."
            )

        if rpeaks is None:
            _, rpeaks = scg_peaks(scg_signals["scg_Clean"], sampling_rate=sampling_rate)

        fig = signal_fixpeaks(rpeaks, sampling_rate=sampling_rate, iterative=True, show=True, method="Kubios")

    return fig
