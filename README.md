# Optimizing contact tracing by identifying influential spreaders through socio-demographic characteristics

### Requirements
- German Socio-Economic Panel (https://www.diw.de/en/diw_01.c.601584.en/data_access.html)
- Anaconda(https://www.anaconda.com/download)
- R

### Preparation
1. Clone this repository.
2. Install the conda environment from `env.yaml`
3. Copy the GSOEP-folder "`SOEP-CORE.v**eu_STATA"` into `/data_input/soep/`.
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