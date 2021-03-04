# - * - coding: utf-8 - * -

from ..signal import signal_fixpeaks, signal_formatpeaks
from .scg_findpeaks import scg_findpeaks


def scg_peaks(scg_cleaned, sampling_rate=1000, method="neurokit", correct_artifacts=False):
    """Find R-peaks in an scg signal.

    Find R-peaks in an scg signal using the specified method.

    Parameters
    ----------
    scg_cleaned : Union[list, np.array, pd.Series]
        The cleaned scg channel as returned by `scg_clean()`.
    sampling_rate : int
        The sampling frequency of `scg_signal` (in Hz, i.e., samples/second).
        Defaults to 1000.
    method : string
        The algorithm to be used for R-peak detection. Can be one of 'neurokit' (default), 'pamtompkins1985',
        'hamilton2002', 'christov2004', 'gamboa2008', 'elgendi2010', 'engzeemod2012' or 'kalidas2017'.
    correct_artifacts : bool
        Whether or not to identify artifacts as defined by Jukka A. Lipponen & Mika P. Tarvainen (2019):
        A robust algorithm for heart rate variability time series artefact correction using novel beat
        classification, Journal of Medical Engineering & Technology, DOI: 10.1080/03091902.2019.1640306.

    Returns
    -------
    signals : DataFrame
        A DataFrame of same length as the input signal in which occurences of R-peaks marked as "1"
        in a list of zeros with the same length as `scg_cleaned`. Accessible with the keys "scg_R_Peaks".
    info : dict
        A dictionary containing additional information, in this case the samples at which R-peaks occur,
        accessible with the key "scg_R_Peaks".

    See Also
    --------
    scg_clean, scg_findpeaks, scg_process, scg_plot, signal_rate,
    signal_fixpeaks

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> scg = nk.scg_simulate(duration=10, sampling_rate=1000)
    >>> cleaned = nk.scg_clean(scg, sampling_rate=1000)
    >>> signals, info = nk.scg_peaks(cleaned, correct_artifacts=True)
    >>> nk.events_plot(info["scg_R_Peaks"], cleaned) #doctest: +ELLIPSIS
    <Figure ...>

    References
    ----------
    - Gamboa, H. (2008). Multi-modal behavioral biometrics based on hci and electrophysiology.
      PhD ThesisUniversidade.

    - W. Zong, T. Heldt, G.B. Moody, and R.G. Mark. An open-source algorithm to detect onset of arterial
      blood pressure pulses. In Computers in Cardiology, 2003, pages 259–262, 2003.

    - Hamilton, Open Source scg Analysis Software Documentation, E.P.Limited, 2002.

    - Jiapu Pan and Willis J. Tompkins. A Real-Time QRS Detection Algorithm. In: IEEE Transactions on
      Biomedical Engineering BME-32.3 (1985), pp. 230–236.

    - C. Zeelenberg, A single scan algorithm for QRS detection and feature extraction, IEEE Comp. in
      Cardiology, vol. 6, pp. 37-42, 1979

    - A. Lourenco, H. Silva, P. Leite, R. Lourenco and A. Fred, "Real Time Electrocardiogram Segmentation
      for Finger Based scg Biometrics", BIOSIGNALS 2012, pp. 49-54, 2012.

    """
    rpeaks = scg_findpeaks(scg_cleaned, sampling_rate=sampling_rate, method=method)

    if correct_artifacts:
        _, rpeaks = signal_fixpeaks(rpeaks, sampling_rate=sampling_rate, iterative=True, method="Kubios")

        rpeaks = {"scg_R_Peaks": rpeaks}

    instant_peaks = signal_formatpeaks(rpeaks, desired_length=len(scg_cleaned), peak_indices=rpeaks)
    signals = instant_peaks
    info = rpeaks

    return signals, info
