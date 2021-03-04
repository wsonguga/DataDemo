"""Submodule for NeuroKit."""

# Aliases
from ..signal import signal_rate as scg_rate
from .scg_analyze import scg_analyze
from .scg_clean import scg_clean
from .scg_delineate import scg_delineate
from .scg_eventrelated import scg_eventrelated
from .scg_findpeaks import scg_findpeaks
from .scg_intervalrelated import scg_intervalrelated
from .scg_peaks import scg_peaks
from .scg_phase import scg_phase
from .scg_plot import scg_plot
from .scg_process import scg_process
from .scg_quality import scg_quality
from .scg_rsp import scg_rsp
from .scg_segment import scg_segment
from .scg_simulate import scg_simulate


__all__ = [
    "scg_simulate",
    "scg_clean",
    "scg_findpeaks",
    "scg_peaks",
    "scg_segment",
    "scg_process",
    "scg_plot",
    "scg_delineate",
    "scg_rsp",
    "scg_phase",
    "scg_quality",
    "scg_eventrelated",
    "scg_intervalrelated",
    "scg_analyze",
    "scg_rate",
]
