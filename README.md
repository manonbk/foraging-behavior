*Study or paramecium foraging behavior*

This repository help analysing data from paramecium recordings in Romain BRETTE's lab, ISIR

It is organized as follows

- folder preprocess helps to clean the raw detection and tracking data (the yolo output)
- folders interface (for interface tools) and trajectories (for trajectory analysis tools) are from romain_brett/Paramecium_behavior
- folder behavior_analysis contains some spatial and temporal empirical analysis of swim/stop behaviors
- folder HMM helps build a hidden markov model of the swim/stop behavior

## Installation

From the repository root, create an environment and install the Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The upstream `movie` module is not included in this repository. Add that module to
your Python environment separately to use movie-opening and video-annotation
features. Tracking-file selection and preprocessing do not import it unless a
movie is opened.

Both HMM scripts additionally require `hmmlearn`. Install it with
`python -m pip install -r requirements-hmm.txt`. On Windows, `hmmlearn` builds a
native extension, so Microsoft C++ Build Tools may be needed.

Trackpy and Norfair are optional integrations for trajectory linking. Install
them separately with `python -m pip install trackpy norfair` if you use those
features.

## Running

Run package scripts from the repository root with `python -m`, for example:

```powershell
python -m preprocess.preprocess_single_cell
python -m behavior_analysis.A01_swim_stop
python -m behavior_analysis.A02_durations
python -m HMM.HMM_fit
```

`A01_swim_stop` writes the tables consumed by the later behavior-analysis scripts
to the selected recording's `movement_analysis` directory.