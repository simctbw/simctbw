from src.entities.agent import PandemicAgent
import src.research.def_population as def_population
from src.entities.model import Model

import pandas as pd

from pathlib import Path
import datetime as dt
import os

from src.logging import log


def inc_to_rel(incidence):
    return incidence / 100000


def model_func(
    master_model_index,
    run_folder_path,
    mode="run",  # "calib"
    output_type="model_output",
    trial=None,
    n_replications=50,
    n_agents=10000,
    name_of_run=None,
    degree=True,
    lockdown_population_sort_mode=None,
    lockdown_frac=None,
    vaccination_population_sort_mode=None,
    vaccination_frac=None,
    vaccination_efficacy=None,
    contact_tracing_population_sort_mode=None,
    contact_tracing_frac=None,
    contact_tracing_contact_selection_mode=None,
    contact_tracing_contact_selection_frac=None,
    contact_tracing_detected_infections=None,
    ct_start_date=None,
    n_days_contact_tracing_isolation=None,
    contact_tracing_max_currently_traced=None,
    population_selection_mode=None,
    stop_sim_if_0_infections=False,
    n_days_save_contact_network=0,
    inf_p_shift=0,
    ct_isolation_level=0,
    scaling_factor=1,
    save_only_these_columns=None,
    df_agents_in_households=None,
    duration_infectious=2,
    start_date=dt.date(2022, 6, 1),
    end_date=dt.date(2022, 8, 1),
):
    log.warning(f"Starte model_func {master_model_index}")

    params = {}
    params["n_agents"] = n_agents
    params["weight"] = True
    params["timetable"] = False
    params["inf_p_shift"] = inf_p_shift

    age_specific_random_initial_infections = True
    n_days_save_contact_network = n_days_save_contact_network
    infection_probability = 0.331194868067963 + inf_p_shift
    lin_age_inf_p_adj = None
    start_date = start_date
    end_date = end_date if end_date is not None else dt.date(2022, 8, 1)
    location_declarants = def_population.def_locations()
    degree = False

    if mode == "eval":
        n_days_save_contact_network = 7
        degree = True

    elif mode == "calib":
        params["infection_probability"] = trial.suggest_float(
            "infection_probability", 0.32, 0.34
        )
        params["lin_age_inf_p_adj"] = None
        lin_age_inf_p_adj = params["lin_age_inf_p_adj"]
        n_days_save_contact_network = 0

    model = Model(
        run_folder_path=run_folder_path,
        scaling_factor=scaling_factor,
        mode=mode,
        degree=degree,
        start_date=start_date,
        end_date=end_date,
        df_agents_in_households=(
            def_population.get_soep4sim(year=2018, area="BW")
            if df_agents_in_households is None
            else df_agents_in_households
        ),
        location_declarants=location_declarants,
        n_agents=params["n_agents"],
        timetable=(def_population.timetable_bw_2022 if params["timetable"] else None),
        initial_infections=(
            {
                "attribute": "age_group_rki",  #  Date: 1.June 2022; Source: https://zenodo.org/record/8307378
                0: inc_to_rel(60),
                5: inc_to_rel(151),
                15: inc_to_rel(256),
                35: inc_to_rel(227),
                60: inc_to_rel(132),
                80: inc_to_rel(79),
            }
            if age_specific_random_initial_infections
            else inc_to_rel(100)
        ),
        n_days_save_contact_network=n_days_save_contact_network,
        stop_sim_if_0_infections=stop_sim_if_0_infections,
        household_weight_column=("weight" if params["weight"] else None),
        infection_probability=infection_probability,
        lin_age_inf_p_adj=lin_age_inf_p_adj,
        ct_isolation_level=ct_isolation_level,
        ct_start_date=ct_start_date,
        replace_population=False,
        n_replications=n_replications,
        output_type=output_type,
        master_model_index=master_model_index,
        folder_path=run_folder_path,
        name_of_run=name_of_run,
        add_info=params,
        agent_class=PandemicAgent,
        lockdown_population_sort_mode=lockdown_population_sort_mode,
        lockdown_frac=lockdown_frac,
        contact_tracing_population_sort_mode=contact_tracing_population_sort_mode,
        contact_tracing_frac=contact_tracing_frac,
        contact_tracing_contact_selection_mode=contact_tracing_contact_selection_mode,
        contact_tracing_contact_selection_frac=contact_tracing_contact_selection_frac,
        contact_tracing_detected_infections=contact_tracing_detected_infections,
        n_days_contact_tracing_isolation=n_days_contact_tracing_isolation,
        contact_tracing_max_currently_traced=contact_tracing_max_currently_traced,
        vaccination_population_sort_mode=vaccination_population_sort_mode,
        vaccination_frac=vaccination_frac,
        vaccination_efficacy=vaccination_efficacy,
        population_selection_mode=population_selection_mode,
        save_only_these_columns=save_only_these_columns,
        duration_infectious=duration_infectious,
    )

    # run the model
    model_output = model.run()

    output_dict = {
        "model_output": model_output,
    }

    if mode == "calib":
        df_params = pd.concat(
            [
                pd.read_csv(
                    Path.joinpath(
                        run_folder_path,
                        f"df_params_{str(master_model_index)}_{str(rep_index)}.csv",
                    )
                )
                for rep_index in range(model.n_replications)
            ]
        )

        rmse_incidence = df_params["rmse_incidence"].mean()
        rmse_cum_infections_by_age = df_params["rmse_cum_infections_by_age"].mean()

        output_dict.update(
            {
                "rmse_incidence": rmse_incidence,
                "rmse_cum_infections_by_age": rmse_cum_infections_by_age,
            }
        )

        del df_params

        for rep_index in range(model.n_replications):
            os.remove(
                Path.joinpath(
                    run_folder_path,
                    f"df_params_{str(master_model_index)}_{str(rep_index)}.csv",
                )
            )

    log.warning(f"Beende model_func {master_model_index}")
    return output_dict
