# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal

from ..misc import as_vector
from ..signal import signal_filter


def scg_clean(scg_signal, sampling_rate=1000, method="neurokit"):
    """Clean an scg signal.

    Prepare a raw scg signal for R-peak detection with the specified method.

    Parameters
    ----------
    scg_signal : Union[list, np.array, pd.Series]
        The raw scg channel.
    sampling_rate : int
        The sampling frequency of `scg_signal` (in Hz, i.e., samples/second).
        Defaults to 1000.
    method : str
        The processing pipeline to apply. Can be one of 'neurokit' (default),
        'biosppy', 'pantompkins1985', 'hamilton2002', 'elgendi2010', 'engzeemod2012'.

    Returns
    -------
    array
        Vector containing the cleaned scg signal.

    See Also
    --------
    scg_findpeaks, signal_rate, scg_process, scg_plot

    Examples
    --------

    >>> import pandas as pd
    >>> import neurokit2 as nk
    >>> import matplotlib.pyplot as plt
    >>>
    >>> scg = nk.scg_simulate(duration=10, sampling_rate=1000)
    >>> signals = pd.DataFrame({"scg_Raw" : scg,
    ...                         "scg_NeuroKit" : nk.scg_clean(scg, sampling_rate=1000, method="neurokit"),
    ...                         "scg_BioSPPy" : nk.scg_clean(scg, sampling_rate=1000, method="biosppy"),
    ...                         "scg_PanTompkins" : nk.scg_clean(scg, sampling_rate=1000, method="pantompkins1985"),
    ...                         "scg_Hamilton" : nk.scg_clean(scg, sampling_rate=1000, method="hamilton2002"),
    ...                         "scg_Elgendi" : nk.scg_clean(scg, sampling_rate=1000, method="elgendi2010"),
    ...                         "scg_EngZeeMod" : nk.scg_clean(scg, sampling_rate=1000, method="engzeemod2012")})
     >>> signals.plot() #doctest: +ELLIPSIS
     <AxesSubplot:>


    References
    --------------
    - Jiapu Pan and Willis J. Tompkins. A Real-Time QRS Detection Algorithm. In: IEEE Transactions on
      Biomedical Engineering BME-32.3 (1985), pp. 230–236.

    - Hamilton, Open Source scg Analysis Software Documentation, E.P.Limited, 2002.

    """
    scg_signal = as_vector(scg_signal)

    method = method.lower()  # remove capitalised letters
    if method in ["nk", "nk2", "neurokit", "neurokit2"]:
        clean = _scg_clean_nk(scg_signal, sampling_rate)
    elif method in ["biosppy", "gamboa2008"]:
        clean = _scg_clean_biosppy(scg_signal, sampling_rate)
    elif method in ["pantompkins", "pantompkins1985"]:
        clean = _scg_clean_pantompkins(scg_signal, sampling_rate)
    elif method in ["hamilton", "hamilton2002"]:
        clean = _scg_clean_hamilton(scg_signal, sampling_rate)
    elif method in ["elgendi", "elgendi2010"]:
        clean = _scg_clean_elgendi(scg_signal, sampling_rate)
    elif method in ["engzee", "engzee2012", "engzeemod", "engzeemod2012"]:
        clean = _scg_clean_engzee(scg_signal, sampling_rate)
    elif method in [
        "christov",
        "christov2004",
        "ssf",
        "slopesumfunction",
        "zong",
        "zong2003",
        "kalidas2017",
        "swt",
        "kalidas",
        "kalidastamil",
        "kalidastamil2017",
    ]:
        clean = scg_signal
    else:
        raise ValueError(
            "NeuroKit error: scg_clean(): 'method' should be "
            "one of 'neurokit', 'biosppy', 'pantompkins1985',"
            " 'hamilton2002', 'elgendi2010', 'engzeemod2012'."
        )
    return clean


# =============================================================================
# Neurokit
# =============================================================================
def _scg_clean_nk(scg_signal, sampling_rate=1000):

    # Remove slow drift and dc offset with highpass Butterworth.
    clean = signal_filter(signal=scg_signal, sampling_rate=sampling_rate, lowcut=0.5, method="butterworth", order=5)

    clean = signal_filter(signal=clean, sampling_rate=sampling_rate, method="powerline", powerline=50)
    return clean


# =============================================================================
# Biosppy
# =============================================================================
def _scg_clean_biosppy(scg_signal, sampling_rate=1000):
    """Adapted from https://github.com/PIA-
    Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/scg.py#L69."""

    order = int(0.3 * sampling_rate)
    if order % 2 == 0:
        order += 1  # Enforce odd number

    # -> filter_signal()
    frequency = [3, 45]

    #   -> get_filter()
    #     -> _norm_freq()
    frequency = 2 * np.array(frequency) / sampling_rate  # Normalize frequency to Nyquist Frequency (Fs/2).

    #     -> get coeffs
    a = np.array([1])
    b = scipy.signal.firwin(numtaps=order, cutoff=frequency, pass_zero=False)

    # _filter_signal()
    filtered = scipy.signal.filtfilt(b, a, scg_signal)

    return filtered


# =============================================================================
# Pan & Tompkins (1985)
# =============================================================================
def _scg_clean_pantompkins(scg_signal, sampling_rate=1000):
    """Adapted from https://github.com/PIA-
    Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/scg.py#L69."""

    f1 = 5 / sampling_rate
    f2 = 15 / sampling_rate
    order = 1

    b, a = scipy.signal.butter(order, [f1 * 2, f2 * 2], btype="bandpass")

    return scipy.signal.lfilter(b, a, scg_signal)  # Return filtered


# =============================================================================
# Elgendi et al. (2010)
# =============================================================================
def _scg_clean_elgendi(scg_signal, sampling_rate=1000):
    """From https://github.com/berndporr/py-scg-detectors/

    - Elgendi, Mohamed & Jonkman, Mirjam & De Boer, Friso. (2010). Frequency Bands Effects on QRS
      Detection. The 3rd International Conference on Bio-inspired Systems and Signal Processing
      (BIOSIGNALS2010). 428-431.

    """

    f1 = 8 / sampling_rate
    f2 = 20 / sampling_rate

    b, a = scipy.signal.butter(2, [f1 * 2, f2 * 2], btype="bandpass")

    return scipy.signal.lfilter(b, a, scg_signal)  # Return filtered


# =============================================================================
# Hamilton (2002)
# =============================================================================
def _scg_clean_hamilton(scg_signal, sampling_rate=1000):
    """Adapted from https://github.com/PIA-
    Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/scg.py#L69."""

    f1 = 8 / sampling_rate
    f2 = 16 / sampling_rate

    b, a = scipy.signal.butter(1, [f1 * 2, f2 * 2], btype="bandpass")

    return scipy.signal.lfilter(b, a, scg_signal)  # Return filtered


# =============================================================================
# Engzee Modified (2012)
# =============================================================================
def _scg_clean_engzee(scg_signal, sampling_rate=1000):
    """From https://github.com/berndporr/py-scg-detectors/

    - C. Zeelenberg, A single scan algorithm for QRS detection and feature extraction, IEEE Comp.
      in Cardiology, vol. 6, pp. 37-42, 1979.

    - A. Lourenco, H. Silva, P. Leite, R. Lourenco and A. Fred, "Real Time Electrocardiogram Segmentation
      for Finger Based scg Biometrics", BIOSIGNALS 2012, pp. 49-54, 2012.

    """

    f1 = 48 / sampling_rate
    f2 = 52 / sampling_rate
    b, a = scipy.signal.butter(4, [f1 * 2, f2 * 2], btype="bandstop")
    return scipy.signal.lfilter(b, a, scg_signal)  # Return filtered
