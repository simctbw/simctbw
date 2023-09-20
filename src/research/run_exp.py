import src
import src.entities.utils as utils
from src.research.def_model import model_func

import itertools
import os
from pathlib import Path
import sys

from mpire import WorkerPool
import pandas as pd


##############################################################################
# INPUT CHECK
##############################################################################

experiments = ["vaccination", "lockdown", "contact_tracing", "baseline"]

if len(sys.argv) > 1:
    experiment = sys.argv[1]
    assert experiment in experiments, f"Experiment must be on of{experiments}"
else:
    experiment = "baseline"

NAME_OF_RUN = sys.argv[2]


##############################################################################
# PARAMETERS
##############################################################################

N_BASELINE_MODELS = 10
N_REPS_PER_MODEL =  50 #In exp: 50   #In sens: 2 * 5
N_REPS_PER_PARAMS = 4  #in exp: 2*4  #In sens: 1
N_CORES = 60
N_AGENTS = 10000

# defines how the population is sorted before a specific proportion of agents gets selected
population_sort_modes = [
    "random",
    "super-spreaders",
    "old",
    "young",
    "age",
    "household_size",
    "inc_rank_emp",
    "inc_rank_sim",
    ]

# the proportion of agents that get selected from the population (for instance, to get vaccinated)
population_fracs = [
    0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    #0.6,
    #0.7,
    #0.8,
    #0.9,
    #1,
    ]

population_frac_selection_modes = [
    "individuals",
    #"households_of_individuals",
]

# the probability of avoiding the virus due to vaccination if the agent would normally have been infected
vaccination_efficacies = [
    0.33,
    0.66,
    0.99,
    ]

# how the contacts of an contact tracing agent are sorted before the proportion of contacts which should be traced are selected
contact_tracing_contact_selection_modes = [ 
    #"random", 
    "weight",
    #"pred_value",
    #"old",
    #"young",
    #"weighted_pred_value",
]

# the proportion of contacts of a contact tracing agent that is selected to getting traced
contact_tracing_contact_selection_fracs = [
    #0.6,
    0.8,
    #1,
]

# Whose contacts should be traced? Only the contacts of symptomatic cases? Or those of all infected agents (that are selected for contact tracing)?
contact_tracing_detected_infections = [
    "ill_symptomatic",
    #"ill",
]

n_days_contact_tracing_isolation = [
    #5,
    10,
    #15,
]

inf_p_shift = [
    #-0.05,
     0.0,
     #0.05,
]


##############################################################################
# CREATE PARAMETER COMBINATIONS
##############################################################################

def generate_params(
        params1=[None], 
        params2=[None], 
        params3=[None], 
        params4=[None], 
        params5=[None], 
        params6=[None],
        params7=[None],
        params8=[None],
        add_i=True, 
        n_reps=1,
        ):
    params = list(itertools.product(params1, params2, params3, params4, params5, params6, params7, params8))
    params = [list(p_list) for p_list in params]

    rep_params = []
    for i in range(n_reps):
        for p_list in params:
            rep_params.append(p_list[:])

    if add_i:
        for i, p_list in enumerate(rep_params):
            p_list.insert(0, i)
    
    return rep_params

# vaccination scenario
if experiment == "vaccination":
    params = generate_params(
        params1=population_sort_modes, 
        params2=population_fracs, 
        params3=population_frac_selection_modes, 
        params4=vaccination_efficacies,
        n_reps=N_REPS_PER_PARAMS,
        )

# lockdown scenario
elif experiment == "lockdown":
    params = generate_params(
        params1=population_sort_modes, 
        params2=population_fracs, 
        params3=population_frac_selection_modes,
        n_reps=N_REPS_PER_PARAMS,
        )

# contact tracing scenario
elif experiment == "contact_tracing":
    params = generate_params(
        params1=population_sort_modes, 
        params2=population_fracs,
        params3=population_frac_selection_modes,
        params4=contact_tracing_contact_selection_modes, 
        params5=contact_tracing_contact_selection_fracs, 
        params6=contact_tracing_detected_infections,
        params7=n_days_contact_tracing_isolation,
        params8=inf_p_shift,
        n_reps=N_REPS_PER_PARAMS,
        )

# no intervention scenario
elif experiment == "baseline":
    none_list = [None] * N_BASELINE_MODELS
    params = generate_params(none_list, n_reps=1)

# invalid scenario-name
else:
    raise ValueError


##############################################################################
# CREATE FOLDER PATH
##############################################################################

def create_run_folder_path(name_of_run):
    return Path.joinpath(src.PATH, "data_output", "run_" + (utils.str_time() if name_of_run is None else name_of_run))

run_folder_path = create_run_folder_path(NAME_OF_RUN)


##############################################################################
# CREATE FOLDER
##############################################################################

def distributed_experiment(i, population_sort_mode, population_frac, population_frac_selection_mode, p4, p5, p6, p7, p8):
    """Helps to run the simulation in parallel."""

    #output_dict = model_func(
    model_func(
        master_model_index="-".join(
        ["m"+str(i), 
         str(population_sort_mode), 
         str(population_frac), 
         str(population_frac_selection_mode), 
         str(p4), 
         str(p5), 
         str(p6), 
         str(p7),
         str(p8),
         ]
        ), 
        run_folder_path=run_folder_path, 
        output_type="save_rep_on_disk", 
        mode="run",
        n_agents=N_AGENTS,
        n_replications=N_REPS_PER_MODEL,
        name_of_run=NAME_OF_RUN,
        population_selection_mode = (population_frac_selection_mode if experiment in ["vaccination", "lockdown", "contact_tracing"] else None),
        vaccination_population_sort_mode = (population_sort_mode if experiment == "vaccination" else None),
        vaccination_frac = (population_frac if experiment == "vaccination" else None),
        vaccination_efficacy = (p4 if experiment == "vaccination" else None),
        lockdown_population_sort_mode = (population_sort_mode if experiment == "lockdown" else None),
        lockdown_frac = (population_frac if experiment == "lockdown" else None),
        contact_tracing_population_sort_mode = (population_sort_mode if experiment == "contact_tracing" else None),
        contact_tracing_frac = (population_frac if experiment == "contact_tracing" else None),
        contact_tracing_contact_selection_mode = (p4 if experiment == "contact_tracing" else None),
        contact_tracing_contact_selection_frac = (p5 if experiment == "contact_tracing" else None),
        contact_tracing_detected_infections = (p6 if experiment == "contact_tracing" else None),
        n_days_contact_tracing_isolation = (p7 if experiment == "contact_tracing" else None),
        inf_p_shift = (p8 if experiment == "contact_tracing" else 0),
        n_days_save_contact_network=(0 if experiment != "baseline" else 7),
        stop_sim_if_0_infections=True,
        save_only_these_columns = ([
            "name_of_run",
            "master_model_index",
            "replication",
            "unique_agent_id",
            "unique_household_id",
            "id",
            "household_id",
            #"vaccination_population_sort_mode", 
            #"vaccination_frac", 
            #"vaccination_efficacy", 
            "population_selection_mode",
            #"lockdown_population_sort_mode",
            #"lockdown_frac",
            "contact_tracing_population_sort_mode",
            "contact_tracing_frac",
            "contact_tracing_contact_selection_mode",
            "contact_tracing_contact_selection_frac",
            "contact_tracing_detected_infections",
            "ever_infected",
            "date_infection",
            "date_icu",
            "age",
            "patient0",
            #"vaccinated",
            "n_times_contact_traced",
            "n_contacts_traced",
            "n_times_contact_tracing",
            "contact_tracing_dates_list",
            "n_days_contact_tracing_isolation",
            ] if experiment != "baseline" else None),
        )


##############################################################################
# RUN EXPERIMENT
##############################################################################

def run():
    if os.path.exists(run_folder_path):
        raise ValueError ("This name of run already exists.")
    else:
        os.mkdir(run_folder_path)
    
    with WorkerPool(n_jobs=N_CORES, use_dill=False) as pool:
        pool.map(
            distributed_experiment, 
            params, 
            chunk_size=1,
            worker_lifespan=1,
            progress_bar=True,
            )
    
    df = pd.concat([pd.read_csv(path) for path in Path.iterdir(run_folder_path) if "agents" in str(path)])
    
    for path in Path.iterdir(run_folder_path):
        if "agents" in str(path):
            os.remove(path)

    for path in Path.iterdir(run_folder_path):
        if "params" in str(path):
            os.remove(path)
    
    df.to_csv(Path.joinpath(run_folder_path, f"df_agents_{str(NAME_OF_RUN)}.csv"), index=False)
    
    del(df)
    

if __name__ == "__main__":
    run()