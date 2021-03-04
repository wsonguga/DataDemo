# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..signal import signal_phase
from .scg_delineate import scg_delineate
from .scg_peaks import scg_peaks


def scg_phase(scg_cleaned, rpeaks=None, delineate_info=None, sampling_rate=None):
    """Compute cardiac phase (for both atrial and ventricular).

    Finds the cardiac phase, labelled as 1 for systole and 0 for diastole.

    Parameters
    ----------
    scg_cleaned : Union[list, np.array, pd.Series]
        The cleaned scg channel as returned by `scg_clean()`.
    rpeaks : list or array or DataFrame or Series or dict
        The samples at which the different scg peaks occur. If a dict or a DataFrame is passed, it is
        assumed that these containers were obtained with `scg_findpeaks()` or `scg_peaks()`.
    delineate_info : dict
        A dictionary containing additional information of scg delineation and can be obtained with
        `scg_delineate()`.
    sampling_rate : int
        The sampling frequency of `scg_signal` (in Hz, i.e., samples/second). Defaults to None.

    Returns
    -------
    signals : DataFrame
        A DataFrame of same length as `scg_signal` containing the following
        columns:

        - *"scg_Phase_Atrial"*: cardiac phase, marked by "1" for systole and "0" for diastole.

        - *"scg_Phase_Completion_Atrial"*: cardiac phase (atrial) completion, expressed in percentage
          (from 0 to 1), representing the stage of the current cardiac phase.

        - *"scg_Phase_Ventricular"*: cardiac phase, marked by "1" for systole and "0" for diastole.

        - *"scg_Phase_Completion_Ventricular"*: cardiac phase (ventricular) completion, expressed in
          percentage (from 0 to 1), representing the stage of the current cardiac phase.

    See Also
    --------
    scg_clean, scg_peaks, scg_process, scg_plot

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> scg = nk.scg_simulate(duration=10, sampling_rate=1000)
    >>> cleaned = nk.scg_clean(scg, sampling_rate=1000)
    >>> _, rpeaks = nk.scg_peaks(cleaned)
    >>> signals, waves = nk.scg_delineate(cleaned, rpeaks, sampling_rate=1000)
    >>>
    >>> cardiac_phase = nk.scg_phase(scg_cleaned=cleaned, rpeaks=rpeaks,
    ...                              delineate_info=waves, sampling_rate=1000)
    >>> nk.signal_plot([cleaned, cardiac_phase], standardize=True) #doctest: +ELLIPSIS

    """
    # Sanitize inputs
    if rpeaks is None:
        if sampling_rate is not None:
            _, rpeaks = scg_peaks(scg_cleaned, sampling_rate=sampling_rate)
        else:
            raise ValueError(
                "R-peaks will be obtained using `nk.scg_peaks`. Please provide the sampling_rate of scg_signal."
            )
    # Try retrieving right column
    if isinstance(rpeaks, dict):
        rpeaks = rpeaks["scg_R_Peaks"]

    if delineate_info is None:
        __, delineate_info = scg_delineate(scg_cleaned, sampling_rate=sampling_rate)

    # Try retrieving right column
    if isinstance(delineate_info, dict):    # FIXME: if this evaluates to False, toffsets and ppeaks are not instantiated

        toffsets = np.full(len(scg_cleaned), False, dtype=bool)
        toffsets_idcs = [int(x) for x in delineate_info["scg_T_Offsets"] if ~np.isnan(x)]
        toffsets[toffsets_idcs] = True

        ppeaks = np.full(len(scg_cleaned), False, dtype=bool)
        ppeaks_idcs = [int(x) for x in delineate_info["scg_P_Peaks"] if ~np.isnan(x)]
        ppeaks[ppeaks_idcs] = True

    # Atrial Phase
    atrial = np.full(len(scg_cleaned), np.nan)
    atrial[rpeaks] = 0.0
    atrial[ppeaks] = 1.0

    last_element = np.where(~np.isnan(atrial))[0][-1]  # Avoid filling beyond the last peak/trough
    atrial[0:last_element] = pd.Series(atrial).fillna(method="ffill").values[0:last_element]

    # Atrial Phase Completion
    atrial_completion = signal_phase(atrial, method="percent")

    # Ventricular Phase
    ventricular = np.full(len(scg_cleaned), np.nan)
    ventricular[toffsets] = 0.0
    ventricular[rpeaks] = 1.0

    last_element = np.where(~np.isnan(ventricular))[0][-1]  # Avoid filling beyond the last peak/trough
    ventricular[0:last_element] = pd.Series(ventricular).fillna(method="ffill").values[0:last_element]

    # Ventricular Phase Completion
    ventricular_comletion = signal_phase(ventricular, method="percent")

    return pd.DataFrame(
        {
            "scg_Phase_Atrial": atrial,
            "scg_Phase_Completion_Atrial": atrial_completion,
            "scg_Phase_Ventricular": ventricular,
            "scg_Phase_Completion_Ventricular": ventricular_comletion,
        }
    )
