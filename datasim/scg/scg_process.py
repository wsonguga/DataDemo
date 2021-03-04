# -*- coding: utf-8 -*-
import pandas as pd

from ..signal import signal_rate, signal_sanitize
from .scg_clean import scg_clean
from .scg_delineate import scg_delineate
from .scg_peaks import scg_peaks
from .scg_phase import scg_phase
from .scg_quality import scg_quality


def scg_process(scg_signal, sampling_rate=1000, method="neurokit"):
    """Process an scg signal.

    Convenience function that automatically processes an scg signal.

    Parameters
    ----------
    scg_signal : Union[list, np.array, pd.Series]
        The raw scg channel.
    sampling_rate : int
        The sampling frequency of `scg_signal` (in Hz, i.e., samples/second).
        Defaults to 1000.
    method : str
        The processing pipeline to apply. Defaults to "neurokit".

    Returns
    -------
    signals : DataFrame
        A DataFrame of the same length as the `scg_signal` containing the following columns:

        - *"scg_Raw"*: the raw signal.

        - *"scg_Clean"*: the cleaned signal.

        - *"scg_R_Peaks"*: the R-peaks marked as "1" in a list of zeros.

        - *"scg_Rate"*: heart rate interpolated between R-peaks.

        - *"scg_P_Peaks"*: the P-peaks marked as "1" in a list of zeros

        - *"scg_Q_Peaks"*: the Q-peaks marked as "1" in a list of zeros .

        - *"scg_S_Peaks"*: the S-peaks marked as "1" in a list of zeros.

        - *"scg_T_Peaks"*: the T-peaks marked as "1" in a list of zeros.

        - *"scg_P_Onsets"*: the P-onsets marked as "1" in a list of zeros.

        - *"scg_P_Offsets"*: the P-offsets marked as "1" in a list of zeros
                            (only when method in `scg_delineate` is wavelet).

        - *"scg_T_Onsets"*: the T-onsets marked as "1" in a list of zeros
                            (only when method in `scg_delineate` is wavelet).

        - *"scg_T_Offsets"*: the T-offsets marked as "1" in a list of zeros.

        - *"scg_R_Onsets"*: the R-onsets marked as "1" in a list of zeros
                            (only when method in `scg_delineate` is wavelet).

        - *"scg_R_Offsets"*: the R-offsets marked as "1" in a list of zeros
                            (only when method in `scg_delineate` is wavelet).

        - *"scg_Phase_Atrial"*: cardiac phase, marked by "1" for systole
          and "0" for diastole.

        - *"scg_Phase_Ventricular"*: cardiac phase, marked by "1" for systole and "0" for diastole.

        - *"scg_Atrial_PhaseCompletion"*: cardiac phase (atrial) completion, expressed in percentage
          (from 0 to 1), representing the stage of the current cardiac phase.

        - *"scg_Ventricular_PhaseCompletion"*: cardiac phase (ventricular) completion, expressed in
          percentage (from 0 to 1), representing the stage of the current cardiac phase.
    info : dict
        A dictionary containing the samples at which the R-peaks occur, accessible with the key
        "scg_Peaks".

    See Also
    --------
    scg_clean, scg_findpeaks, scg_plot, signal_rate, signal_fixpeaks

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> scg = nk.scg_simulate(duration=15, sampling_rate=1000, heart_rate=80)
    >>> signals, info = nk.scg_process(scg, sampling_rate=1000)
    >>> nk.scg_plot(signals) #doctest: +ELLIPSIS
    <Figure ...>

    """
    # Sanitize input
    scg_signal = signal_sanitize(scg_signal)

    scg_cleaned = scg_clean(scg_signal, sampling_rate=sampling_rate, method=method)
    # R-peaks
    instant_peaks, rpeaks, = scg_peaks(
        scg_cleaned=scg_cleaned, sampling_rate=sampling_rate, method=method, correct_artifacts=True
    )

    rate = signal_rate(rpeaks, sampling_rate=sampling_rate, desired_length=len(scg_cleaned))

    quality = scg_quality(scg_cleaned, rpeaks=None, sampling_rate=sampling_rate)

    signals = pd.DataFrame({"scg_Raw": scg_signal, "scg_Clean": scg_cleaned, "scg_Rate": rate, "scg_Quality": quality})

    # Additional info of the scg signal
    delineate_signal, delineate_info = scg_delineate(
        scg_cleaned=scg_cleaned, rpeaks=rpeaks, sampling_rate=sampling_rate
    )

    cardiac_phase = scg_phase(scg_cleaned=scg_cleaned, rpeaks=rpeaks, delineate_info=delineate_info)

    signals = pd.concat([signals, instant_peaks, delineate_signal, cardiac_phase], axis=1)

    info = rpeaks
    return signals, info
