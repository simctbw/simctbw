# SIMCTBW

This repository contains the code and data of the agent-based simulation model used in the study *Optimizing contact tracing by identifying influential spreaders through socio-demographic characteristics*. To run the simulation, the data of the German Socio-Economic Panel is required. In addition, especially the simulation experiments need many computation resources. However, the data analysis presented in the paper can be partly replicated on a *standard* computer with the preprocessed data in this repository.

### Requirements
- **German Socio-Economic Panel (GSOEP)** - The GSOEP data is required to run the simulation. However, we are not allowed to share the data. Thus, you must request the data directly from DIW Berlin: https://www.diw.de/en/diw_01.c.601584.en/data_access.html.
- **Anaconda**: The primary model code is written in Python and is best installed using the conda environment defined in `env.yaml`. Download Anaconda here: https://www.anaconda.com/download.
- **R**: The data analysis is done in R.
- **Computational resources**: In order to run the simulation and process the output data in an acceptable amount of time, a computer (better: HPC cluster) with at least 124 GB RAM and 16 CPU cores is recommended. In addition, at least 500GB hard disk memory should be available.

### Important files
These are the most important files to understand the model dynamics:
- Main model entities `src/entities/`
    - Model: `src/entities/model.py`
    - Agents: `src/entities/agent.py`
    - Locations: `src/entities/location.py`
- Parameter settings of simulation experiments: `src/research/run_exp.py`
- Further parameter setting for simulation: `src/research/def_model.py`
- Location definitions: `src/research/def_population.py`
- GSOEP preprocessing: `src/research/def_population.py`
- Validation of contact patterns: `src/research/eval_calibration.Rmd`
- Validation of infection dynamics: `src/research/eval_calibration.Rmd`
- Negative binomial regression analysis: `src/research/eval_reg.Rmd`
- Analysis of contact tracing experiments: `src/research/eval_ct.Rmd`
- Sensitivity analysis: `src/research/eval_ct_sens.Rmd`

### Setup
1. Clone this repository.
2. Install the conda environment from `env.yaml` by executing `conda env create -f env.yaml` from the command line.
3. Activate the conda environment by executing `conda activate simctbw_revision` from the command line.
4. Copy the GSOEP-folder `SOEP-CORE.v**eu_STATA` into `/data_input/soep/`.

### Simulation
1. **Validation**: Run the code in `src/research/run_eval.py` to check if everything works as well as to validate the contact patterns and infection dynamics.
2. **Baseline model**: To generate the data on which the regression analysis relies, run `conda run -n simctbw python run_exp.py baseline baseline` from the command line.
3. **Contact tracing experiments**: To conduct the simulation experiments, run `conda run -n simctbw python run_exp.py contact_tracing ct_exp` from the commad line. The scenarios of the experiment are defined in `src/research/run_exp.py`.

### Data analysis
- Validation contact patterns: `src/research/eval_calibration.Rmd`.
- Validation infection dynamics: `src/research/eval_calibration.Rmd`.
- Regression analysis: `src/research/eval_reg.Rmd`. The *baseline model* must be executed first because we cannot share the micro-level data needed for this analysis.
- Contact tracing experiments: `src/research/eval_ct.Rmd`.
