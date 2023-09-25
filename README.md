# SIMCTBW

This repository contains the code and data of the agent-based simulation model used in the study *Optimizing contact tracing by identifying influential spreaders through socio-demographic characteristics*. To run the simulation, the data of the German Socio-Economic Panel is required. In addition, especially the simulation experiments need many computation resources. However, a replication of the data analysis presented in the paper can be done on a *standard* computer with the preprocessed data in this repository.

### Requirements
- **German Socio-Economic Panel (GSOEP)** - The GSOEP data is required to run the simulation. However, we are not allowed to share the data. Thus, you must request the data directly from DIW Berlin: https://www.diw.de/en/diw_01.c.601584.en/data_access.html. The GSOEP is not required if you just want to replicate the data analysis presented in the paper.
- **Anaconda**: The primary model code is written in Python and is best installed using the conda environment defined in `env.yaml`. Download Anaconda here: https://www.anaconda.com/download.
- **R**: The data analysis is done in R.
- **Computational resources**: In order to run the simulation and process the output data in an acceptable amount of time, a computer (better: HPC cluster) with at least 32 (better: 256) GB RAM and 16 (better: 64) is recommended. In addition, at least 120GB hard disk memory should be available.

### Important files
- Main model entities `src/entities/`
    - Model: `src/entities/model.py`
    - Agents: `src/entities/agent.py`
    - Locations: `src/entities/location.py`
- Parameter setting for simulation experiments: `src/research/def_model.py`
- Location definitions: `src/research/def_population.py`
- GSOEP preprocessing: `src/research/def_population.py`
- Validation of contact patterns: `src/research/eval_contacts.Rmd`
- Validation of infection dynamics: `src/research/eval_incidence.Rmd`
- Negative binomial regression analysis: `src/research/eval_reg.Rmd`
- Analysis of contact tracing experiments: `src/research/eval_ct.Rmd`

### Setup
1. Clone this repository.
2. Install the conda environment from `env.yaml` by executing `conda env create -f env.yaml` from the command line.
3. Activate the conda environment by executing `conda activate simctbw` from the command line.
4. Copy the GSOEP-folder `SOEP-CORE.v**eu_STATA` into `/data_input/soep/`.
5. Run all cells in `src/research/eval_model.ipynb` to check if everything works.

### Simulation
- Baseline model for regression analysis: run `conda run -n simctbw python run_exp.py baseline baseline` from the command line.
- Contact tracing experiments: run `conda run -n simctbw python run_exp.py contact_tracing ct_exp` from the commad line.

### Data analysis
- Validation contact patterns: `src/research/eval_contacts.Rmd`
- Validation infection dynamics: `src/research/eval_incidence.Rmd`
- Regression analysis: `src/research/eval_reg.Rmd`
- Contact tracing experiments: `src/research/eval_ct.Rmd`