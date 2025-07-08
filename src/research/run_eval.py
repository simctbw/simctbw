from src.research.def_model import model_func
import src
import src.entities.utils as utils

from pathlib import Path
import time

import random

# set seed
random.seed(0)

NAME_OF_RUN = "eval"
N_REPS = 50
N_AGENTS = 10000
SAVE_DATA = True
SHOW_PLOTS = True

print(1)
# run model
start_time = time.time()
run_folder_path = Path.joinpath(src.PATH, "data_output", (utils.str_time() if NAME_OF_RUN is None else NAME_OF_RUN))

model_output = model_func(
    master_model_index=0, 
    run_folder_path=run_folder_path, 
    mode="eval", 
    n_replications=N_REPS,
    n_agents=N_AGENTS,
    scaling_factor=1,
    )["model_output"]

df_agents = model_output.get_df_agents()
df_agents.to_csv(Path.joinpath(src.PATH, "data_output", "eval", "df_eval_agents.csv"), index=False)

print(1.5)
# eval incidence over time
inc_eval = model_output.eval_incidence(
    sim_date_col="date_symptomatic", 
    per_replication=True,
    plot=SHOW_PLOTS,
    )
df_inc = inc_eval["df_incidence"]
if SAVE_DATA:
    df_inc.to_csv(Path.joinpath(src.PATH, "data_output", "eval", "df_eval_infections_inc7.csv"), index=False)
print(2)

# eval infections per age group
eval_cum_infections = model_output.eval_cum_infections_by_age(
    "ever_infected_with_symptoms", 
    emp_start_date=model_output.start_date, 
    emp_end_date=model_output.end_date, 
    plot_4=False,
    age_specific=False,
    plot=SHOW_PLOTS,
    )
df_cum_infections_simple = eval_cum_infections["df"]
if SAVE_DATA:
    df_cum_infections_simple.to_csv(Path.joinpath(src.PATH, "data_output", "eval", "df_eval_infections_cum_infections_rel.csv"), index=False)
print(3)

# eval incidences within age groups
eval_cum_infections = model_output.eval_cum_infections_by_age(
    "ever_infected_with_symptoms", 
    emp_start_date=model_output.start_date, 
    emp_end_date=model_output.end_date, 
    plot_4=False,
    age_specific=True,
    plot=SHOW_PLOTS,
    )
df_age_infections = eval_cum_infections["df"]
if SAVE_DATA:
    df_age_infections.to_csv(Path.joinpath(src.PATH, "data_output", "eval", "df_eval_infections_cum_infections_age_rel.csv"), index=False)
print(4)
# eval contacts
model_output.eval_age_contact_matrix(weighted_sim_contacts=False, same_color_scale=True, save_data=SAVE_DATA, plot=SHOW_PLOTS)