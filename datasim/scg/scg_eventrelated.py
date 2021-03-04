# -*- coding: utf-8 -*-
from warnings import warn

from ..epochs.eventrelated_utils import (
    _eventrelated_addinfo,
    _eventrelated_rate,
    _eventrelated_sanitizeinput,
    _eventrelated_sanitizeoutput,
)
from ..misc import NeuroKitWarning


def scg_eventrelated(epochs, silent=False):
    """Performs event-related scg analysis on epochs.

    Parameters
    ----------
    epochs : Union[dict, pd.DataFrame]
        A dict containing one DataFrame per event/trial,
        usually obtained via `epochs_create()`, or a DataFrame
        containing all epochs, usually obtained via `epochs_to_df()`.
    silent : bool
        If True, silence possible warnings.

    Returns
    -------
    DataFrame
        A dataframe containing the analyzed scg features for each epoch, with each epoch indicated by
        the `Label` column (if not present, by the `Index` column). The analyzed features consist of
        the following:

        - *"scg_Rate_Max"*: the maximum heart rate after stimulus onset.

        - *"scg_Rate_Min"*: the minimum heart rate after stimulus onset.

        - *"scg_Rate_Mean"*: the mean heart rate after stimulus onset.

        - *"scg_Rate_Max_Time"*: the time at which maximum heart rate occurs.

        - *"scg_Rate_Min_Time"*: the time at which minimum heart rate occurs.

        - *"scg_Phase_Atrial"*: indication of whether the onset of the event concurs with respiratory
          systole (1) or diastole (0).

        - *"scg_Phase_Ventricular"*: indication of whether the onset of the event concurs with respiratory
          systole (1) or diastole (0).

        - *"scg_Phase_Atrial_Completion"*: indication of the stage of the current cardiac (atrial) phase
          (0 to 1) at the onset of the event.

        - *"scg_Phase_Ventricular_Completion"*: indication of the stage of the current cardiac (ventricular)
          phase (0 to 1) at the onset of the event.

        We also include the following *experimental* features related to the parameters of a
        quadratic model:

        - *"scg_Rate_Trend_Linear"*: The parameter corresponding to the linear trend.

        - *"scg_Rate_Trend_Quadratic"*: The parameter corresponding to the curvature.

        - *"scg_Rate_Trend_R2"*: the quality of the quadratic model. If too low, the parameters might
          not be reliable or meaningful.

    See Also
    --------
    events_find, epochs_create, bio_process

    Examples
    ----------
    >>> import neurokit2 as nk
    >>>
    >>> # Example with simulated data
    >>> scg, info = nk.scg_process(nk.scg_simulate(duration=20))
    >>>
    >>> # Process the data
    >>> epochs = nk.epochs_create(scg, events=[5000, 10000, 15000],
    ...                           epochs_start=-0.1, epochs_end=1.9)
    >>> nk.scg_eventrelated(epochs) #doctest: +ELLIPSIS
      Label  Event_Onset  ...  scg_Phase_Completion_Ventricular  scg_Quality_Mean
    1     1          ...  ...                               ...               ...
    2     2          ...  ...                               ...               ...
    3     3          ...  ...                               ...               ...

    [3 rows x 16 columns]
    >>>
    >>> # Example with real data
    >>> data = nk.data("bio_eventrelated_100hz")
    >>>
    >>> # Process the data
    >>> df, info = nk.bio_process(scg=data["scg"], sampling_rate=100)
    >>> events = nk.events_find(data["Photosensor"],
    ...                         threshold_keep='below',
    ...                         event_conditions=["Negative", "Neutral",
    ...                                           "Neutral", "Negative"])
    >>> epochs = nk.epochs_create(df, events, sampling_rate=100,
    ...                           epochs_start=-0.1, epochs_end=1.9)
    >>> nk.scg_eventrelated(epochs) #doctest: +ELLIPSIS
      Label Condition  ...  scg_Phase_Completion_Ventricular  scg_Quality_Mean
    1     1  Negative  ...                               ...               ...
    2     2   Neutral  ...                               ...               ...
    3     3   Neutral  ...                               ...               ...
    4     4  Negative  ...                               ...               ...

    [4 rows x 17 columns]

    """
    # Sanity checks
    epochs = _eventrelated_sanitizeinput(epochs, what="scg", silent=silent)

    # Extract features and build dataframe
    data = {}  # Initialize an empty dict
    for i in epochs.keys():

        data[i] = {}  # Initialize empty container

        # Rate
        data[i] = _eventrelated_rate(epochs[i], data[i], var="scg_Rate")

        # Cardiac Phase
        data[i] = _scg_eventrelated_phase(epochs[i], data[i])

        # Quality
        data[i] = _scg_eventrelated_quality(epochs[i], data[i])

        # Fill with more info
        data[i] = _eventrelated_addinfo(epochs[i], data[i])

    # Return dataframe
    return _eventrelated_sanitizeoutput(data)


# =============================================================================
# Internals
# =============================================================================


def _scg_eventrelated_phase(epoch, output={}):

    # Sanitize input
    if "scg_Phase_Atrial" not in epoch or "scg_Phase_Ventricular" not in epoch:
        warn(
            "Input does not have an `scg_Phase_Artrial` or `scg_Phase_Ventricular` column."
            " Will not indicate whether event onset concurs with cardiac phase.",
            category=NeuroKitWarning
        )
        return output

    # Indication of atrial systole
    output["scg_Phase_Atrial"] = epoch["scg_Phase_Atrial"][epoch.index > 0].iloc[0]
    output["scg_Phase_Completion_Atrial"] = epoch["scg_Phase_Completion_Atrial"][epoch.index > 0].iloc[0]

    # Indication of ventricular systole
    output["scg_Phase_Ventricular"] = epoch["scg_Phase_Ventricular"][epoch.index > 0].iloc[0]
    output["scg_Phase_Completion_Ventricular"] = epoch["scg_Phase_Completion_Ventricular"][epoch.index > 0].iloc[0]

    return output


def _scg_eventrelated_quality(epoch, output={}):

    # Sanitize input
    colnames = epoch.columns.values
    if len([i for i in colnames if "scg_Quality" in i]) == 0:
        warn(
            "Input does not have an `scg_Quality` column."
            " Quality of the signal is not computed.",
            category=NeuroKitWarning
        )
        return output

    # Average signal quality over epochs
    output["scg_Quality_Mean"] = epoch["scg_Quality"].mean()

    return output
