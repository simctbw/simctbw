import src
import src.entities.utils as utils
from src.research.def_model import model_func
import src.research.def_population as def_population

from src.logging import log
import shutil
from tqdm.auto import tqdm

import itertools
import multiprocessing
import os
from pathlib import Path
import sys
import random

from mpire import WorkerPool

import pandas as pd

import datetime as dt

import gc

##############################################################################
# INPUT CHECK
##############################################################################

experiments = ["vaccination", "lockdown", "contact_tracing", "baseline"]

if len(sys.argv) > 1:
    experiment = sys.argv[1]
    assert experiment in experiments, f"Experiment must be one of{experiments}"
else:
    experiment = "baseline"

NAME_OF_RUN = sys.argv[2]


##############################################################################
# PARAMETERS
##############################################################################

USE_MPIRE = False

paramselect = "mainresults"
#paramselect = "long"
#paramselect = "sens"

if paramselect == "mainresults":
    N_BASELINE_MODELS = 30
    N_REPS_PER_MODEL =  50
    N_REPS_PER_PARAMS =  50
    N_CORES = (15 if experiment == "baseline" else 20)
    N_AGENTS = 10000
    START_DATE = dt.date(2022, 6, 1)
    END_DATE = dt.date(2022, 8, 1)
    # 82 hours on 20 cores
    
elif paramselect == "long":
    N_REPS_PER_MODEL = 50
    N_REPS_PER_PARAMS = 10
    N_CORES = 20  
    N_AGENTS = 10000
    START_DATE = dt.date(2022, 6, 1)
    END_DATE = dt.date(2023, 6, 1)

elif paramselect == "sens":
    N_REPS_PER_MODEL = 50
    N_REPS_PER_PARAMS = 25
    N_CORES = 20  
    N_AGENTS = 10000
    START_DATE = dt.date(2022, 6, 1)
    END_DATE = dt.date(2022, 8, 1)

else:
    raise Exception("paramselect is invalid")


# defines how the population is sorted before a specific proportion of agents gets selected
population_sort_modes = [
    "random",
    "super-spreaders",
    "age",
    "household_size",
    "old",
    "young",
    "inc_rank_emp",
    "inc_rank_sim",
]

population_frac_selection_modes = [
    "individuals",
    # "households_of_individuals",
]

# the probability of avoiding the virus due to vaccination if the agent would normally have been infected
vaccination_efficacies = [
    # 0.33,
    0.66,
    # 0.99,
]

# how the contacts of an contact tracing agent are sorted before the proportion of contacts which should be traced are selected
contact_tracing_contact_selection_modes = [  # "random", #"pred_value", #"old", #"young", #"weighted_pred_value",
    "weight",
]

# Whose contacts should be traced? Only the contacts of symptomatic cases? Or those of all infected agents (that are selected for contact tracing)?
contact_tracing_detected_infections = [
    "ill_symptomatic",
]


if paramselect in ("mainresults", "long"):
    population_fracs = [
        0,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
    ]

    # the proportion of contacts of a contact tracing agent that is selected to getting traced
    contact_tracing_contact_selection_fracs = [
        0.8,
    ]

    n_days_contact_tracing_isolation = [
        10,
    ]

    inf_p_shift = [
        0.0,
    ]

    ct_isolation_levels = [
        0.75,
    ]

    ct_max_currently_traced = [
        50,
    ]

    durations_infectious = [
        2,
    ]

    ct_start_dates = [
        dt.date(2022, 6, 1),
    ]

    end_dates = [
        dt.date(2022, 8, 1),
    ]

elif paramselect == "sens":
    population_fracs = [
        0,
        #0.05,
        #0.1,
        0.15,
        #0.2,
        #0.25,
        #0.3,
    ]

    # the proportion of contacts of a contact tracing agent that is selected to getting traced
    contact_tracing_contact_selection_fracs = [
        #0.6,
        0.8,
        #1,
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

    ct_isolation_levels = [
        #0.5,
        0.75,
        #1,
    ]

    ct_max_currently_traced = [
        #25,
        50,
        #75,
    ]


    durations_infectious = [
        #1,
        2,
        #3,
    ]

    ct_start_dates = [
        dt.date(2022, 6, 1),
        #dt.date(2022, 6, 15),
        #dt.date(2022, 7, 1),
    ]

    end_dates = [
        #dt.date(2022, 8, 1),
        dt.date(2022, 12, 1),
        dt.date(2023, 6, 1),
    ]

elif paramselect == "long":
    ct_start_dates = [
        dt.date(2022, 6, 1),
        #dt.date(2022, 7, 1),
        #dt.date(2022, 8, 1),
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
    params9=[None],
    params10=[None],
    params11=[None],
    params12=[None],
    params13=[None],
    add_i=True,
    n_reps=1,
):
    params = list(
        itertools.product(
            params1,
            params2,
            params3,
            params4,
            params5,
            params6,
            params7,
            params8,
            params9,
            params10,
            params11,
            params12,
            params13,
        )
    )
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
        params12=durations_infectious,
        n_reps=N_REPS_PER_PARAMS,
    )

# lockdown scenario
elif experiment == "lockdown":
    params = generate_params(
        params1=population_sort_modes,
        params2=population_fracs,
        params3=population_frac_selection_modes,
        params12=durations_infectious,
        n_reps=N_REPS_PER_PARAMS,
    )

# contact tracing scenario
elif experiment == "contact_tracing":
    if paramselect != "sens":
        params = generate_params(
            params1=population_sort_modes,
            params2=population_fracs,
            params3=population_frac_selection_modes,
            params4=contact_tracing_contact_selection_modes,
            params5=contact_tracing_contact_selection_fracs,
            params6=contact_tracing_detected_infections,
            params7=n_days_contact_tracing_isolation,
            params8=inf_p_shift,
            params9=ct_isolation_levels,
            params10=ct_max_currently_traced,
            params11=ct_start_dates,
            params12=durations_infectious,
            params13=end_dates,
            n_reps=N_REPS_PER_PARAMS,
        )
    
    else:
        def generate_sens_params(
            params1=[None],
            params2=[None],
            params3=[None],
            params4=[None],
            params5=[None],
            params6=[None],
            params7=[None],
            params8=[None],
            params9=[None],
            params10=[None],
            params11=[None],
            params12=[None],
            params13=[None],
            add_i=True,
            n_reps=1,
        ):
            params1 = list(itertools.product(
                params1,
                params2,
                ))
            
            params2 = [
                params3,
                params4,
                params5,
                params6,
                params7,
                params8,
                params9,
                params10,
                params11,
                params12,
                params13,
            ]

            for i, p in enumerate(params2):
                assert len(p) == 1 or len(p) == 3
                if len(p) == 1:
                    p = p * 3
                    params2[i] = p
            
            params2 = [list(p) for p in zip(*params2)]
        
            params2_reps = []
            for i in range(n_reps):
                for p_list in params2:
                    params2_reps.append(p_list[:])

            final_params = []
            for p1 in params1:
                for p2 in params2_reps:
                    p_merged = list(p1).copy()
                    p_merged.extend(p2)
                    final_params.append(p_merged)
            
            if add_i:
                for i, p_list in enumerate(final_params):
                    p_list.insert(0, i)

            return final_params
        
        params = generate_sens_params(
            params1=population_sort_modes,
            params2=population_fracs,
            params3=population_frac_selection_modes,
            params4=contact_tracing_contact_selection_modes,
            params5=contact_tracing_contact_selection_fracs,
            params6=contact_tracing_detected_infections,
            params7=n_days_contact_tracing_isolation,
            params8=inf_p_shift,
            params9=ct_isolation_levels,
            params10=ct_max_currently_traced,
            params11=ct_start_dates,
            params12=durations_infectious,
            params13=end_dates,
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
    return Path.joinpath(
        src.PATH,
        "data_output",
        "run_" + (utils.str_time() if name_of_run is None else name_of_run),
    )


run_folder_path = create_run_folder_path(NAME_OF_RUN)


##############################################################################
# Load data
##############################################################################

soep4sim = def_population.get_soep4sim(year=2018, area="BW")


##############################################################################
# Define function
##############################################################################


def distributed_experiment(params):
    """Helps to run the simulation in parallel."""
    (
        i,
        population_sort_mode,
        population_frac,
        population_frac_selection_mode,
        p4,
        p5,
        p6,
        p7,
        p8,
        p9,
        p10,
        p11,
        p12,
        p13,
    ) = params
    
    try:
        print("Seed:", i)
        random.seed(i)

        # output_dict = model_func(
        model_func(
            master_model_index="-".join(
                [
                    "m" + str(i),
                    str(population_sort_mode),
                    str(population_frac),
                    str(population_frac_selection_mode),
                    str(p4),
                    str(p5),
                    str(p6),
                    str(p7),
                    str(p8),
                    str(p9),
                    str(p10),
                    str(p11),
                    str(p12),
                ]
            ),
            run_folder_path=run_folder_path,
            output_type="save_rep_on_disk",
            mode="run",
            n_agents=N_AGENTS,
            n_replications=N_REPS_PER_MODEL,
            name_of_run=NAME_OF_RUN,
            population_selection_mode=(
                population_frac_selection_mode
                if experiment in ["vaccination", "lockdown", "contact_tracing"]
                else None
            ),
            vaccination_population_sort_mode=(
                population_sort_mode if experiment == "vaccination" else None
            ),
            vaccination_frac=(population_frac if experiment == "vaccination" else None),
            vaccination_efficacy=(p4 if experiment == "vaccination" else None),
            lockdown_population_sort_mode=(
                population_sort_mode if experiment == "lockdown" else None
            ),
            lockdown_frac=(population_frac if experiment == "lockdown" else None),
            contact_tracing_population_sort_mode=(
                population_sort_mode if experiment == "contact_tracing" else None
            ),
            contact_tracing_frac=(
                population_frac if experiment == "contact_tracing" else None
            ),
            contact_tracing_contact_selection_mode=(
                p4 if experiment == "contact_tracing" else None
            ),
            contact_tracing_contact_selection_frac=(
                p5 if experiment == "contact_tracing" else None
            ),
            contact_tracing_detected_infections=(
                p6 if experiment == "contact_tracing" else None
            ),
            n_days_contact_tracing_isolation=(
                p7 if experiment == "contact_tracing" else None
            ),
            inf_p_shift=(p8 if experiment == "contact_tracing" else 0),
            ct_isolation_level=(p9 if experiment == "contact_tracing" else 0),
            ct_start_date=(p11 if experiment == "contact_tracing" else None),
            contact_tracing_max_currently_traced=(
                p10 if experiment == "contact_tracing" else 0
            ),
            n_days_save_contact_network=(0 if experiment != "baseline" else 7),
            duration_infectious=(2 if experiment == "baseline" else p12),
            stop_sim_if_0_infections=True,
            df_agents_in_households=soep4sim,
            start_date=START_DATE,
            end_date=p13,
            save_only_these_columns=(
                [
                    # "name_of_run",
                    "master_model_index",
                    "replication",
                    "unique_agent_id",
                    "unique_household_id",
                    "id",
                    "household_id",
                    # "vaccination_population_sort_mode",
                    # "vaccination_frac",
                    # "vaccination_efficacy",
                    # "population_selection_mode",
                    # "lockdown_population_sort_mode",
                    # "lockdown_frac",
                    "contact_tracing_population_sort_mode",
                    "contact_tracing_frac",
                    # "contact_tracing_contact_selection_mode",
                    "contact_tracing_contact_selection_frac",
                    # "contact_tracing_detected_infections",
                    "ever_infected",
                    "date_infection",
                    "date_icu",
                    "age",
                    "patient0",
                    # "vaccinated",
                    "n_times_contact_traced",
                    "n_contacts_traced",
                    "n_times_contact_tracing",
                    "contact_tracing_dates_list",
                    "n_days_contact_tracing_isolation",
                    "ct_isolation_level",
                    "contact_tracing_max_currently_traced",
                    "ct_start_date",
                    "duration_infectious",
                    "end_date",
                ]
                if experiment != "baseline"
                else None
            ),
        )

    except Exception as e:
        log.exception("Help I died.")
        raise e


##############################################################################
# RUN EXPERIMENT
##############################################################################


def run():
    if os.path.exists(run_folder_path):
        raise ValueError("This name of run already exists.")
    else:
        os.mkdir(run_folder_path)

    if USE_MPIRE:
        with WorkerPool(n_jobs=N_CORES, use_dill=True, enable_insights=True) as pool:
            pool.map(
                distributed_experiment,
                [params],
                chunk_size=1,
                worker_lifespan=1,
                progress_bar=True,
            )
    else:
        with tqdm(total=len(params), disable=False) as pbar:
            with multiprocessing.Pool(processes=N_CORES, maxtasksperchild=1) as pool:
                for result in pool.imap_unordered(
                    distributed_experiment,
                    params,
                    chunksize=1,
                ):
                    pbar.update()

    print("Finished all simulations.")
    gc.collect()


if __name__ == "__main__":
    run()
