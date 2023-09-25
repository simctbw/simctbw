# SIMCTBW

This repository contains the code and data of the agent-based simulation model used in the study *Optimizing contact tracing by identifying influential spreaders through socio-demographic characteristics*.

### Requirements
- **German Socio-Economic Panel (GSOEP)** - The GSOEP data is required to run the simulation. However, we are not allowed to share the data. Thus, you must request the data directly from DIW Berlin: https://www.diw.de/en/diw_01.c.601584.en/data_access.html. The GSOEP is not required if you just want to replicate the data analysis presented in the paper.
- **Anaconda**: The primary model code is written in Python and is best installed using the conda environment defined in `env.yaml`. Download Anaconda here: https://www.anaconda.com/download.
- **R**: The data analysis is done in R.
- **Computational resources**: In order to run the simulation and process the output data in an acceptable amount of time, a computer with at least 32 (better: 256) GB RAM and 16 (better: 64) is recommended. In addition, at least 120GB hard disk memory should be available.

### Setup
1. Clone this repository.
2. Install the conda environment from `env.yaml`
3. Copy the GSOEP-folder `SOEP-CORE.v**eu_STATA` into `/data_input/soep/`.
4. Run all cells in `src/research/eval_model.ipynb`.

### Simulation
- Baseline model for regression analysis: run `conda run -n simctbw python run_exp.py baseline baseline` from the command line.
- Contact tracing experiments: run `conda run -n simctbw python run_exp.py contact_tracing ct_exp` from the commad line.

### Data analysis
- Validation contact patterns: `src/research/eval_contacts.Rmd`
- Validation infection dynamics: `src/research/eval_incidence.Rmd`
- Regression analysis: `src/research/eval_reg.Rmd`
- Contact tracing experiments: `src/research/eval_ct.Rmd`

### Important files
- GSOEP preprocessing and location definition: `src/research/def_population.py`
- Parameter setting for simulation experiments: `src/research/def_model.py`
