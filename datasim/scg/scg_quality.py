# - * - coding: utf-8 - * -
import numpy as np

from ..epochs import epochs_to_df
from ..signal import signal_interpolate
from ..stats import distance, rescale
from .scg_peaks import scg_peaks
from .scg_segment import scg_segment


def scg_quality(scg_cleaned, rpeaks=None, sampling_rate=1000):
    """Quality of scg Signal.

    Compute a continuous index of quality of the scg signal, by interpolating the distance
    of each QRS segment from the average QRS segment present in the data. This index is
    therefore relative, and 1 corresponds to heartbeats that are the closest to the average
    sample and 0 corresponds to the most distance heartbeat, from that average sample.

    Returns
    -------
    array
        Vector containing the quality index ranging from 0 to 1.

    See Also
    --------
    scg_segment, scg_delineate

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> scg = nk.scg_simulate(duration=30, sampling_rate=300, noise=0.2)
    >>> scg_cleaned = nk.scg_clean(scg, sampling_rate=300)
    >>> quality = nk.scg_quality(scg_cleaned, sampling_rate=300)
    >>>
    >>> nk.signal_plot([scg_cleaned, quality], standardize=True)

    """
    # Sanitize inputs
    if rpeaks is None:
        _, rpeaks = scg_peaks(scg_cleaned, sampling_rate=sampling_rate)
        rpeaks = rpeaks["scg_R_Peaks"]

    # Get heartbeats
    heartbeats = scg_segment(scg_cleaned, rpeaks, sampling_rate)
    data = epochs_to_df(heartbeats).pivot(index="Label", columns="Time", values="Signal")
    data.index = data.index.astype(int)
    data = data.sort_index()

    # Filter Nans
    missing = data.T.isnull().sum().values
    nonmissing = np.where(missing == 0)[0]

    data = data.iloc[nonmissing, :]

    # Compute distance
    dist = distance(data, method="mean")
    dist = rescale(np.abs(dist), to=[0, 1])
    dist = np.abs(dist - 1)  # So that 1 is top quality

    # Replace missing by 0
    quality = np.zeros(len(heartbeats))
    quality[nonmissing] = dist

    # Interpolate
    quality = signal_interpolate(rpeaks, quality, x_new=np.arange(len(scg_cleaned)), method="quadratic")

    return quality
