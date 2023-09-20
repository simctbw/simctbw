import src
import src.entities.utils as utils
from src.research.def_model import model_func

from mpire import WorkerPool
import optuna

from pathlib import Path
import sys

if len(sys.argv) > 1 and sys.argv[1] == "test":
    print("test mode")
    N_CORES = 3
    N_TRIALS_PER_CORE = 2
    N_REPS_PER_MODEL = 2
    NAME_OF_RUN = sys.argv[2]
    STUDY_NAME = NAME_OF_RUN
    STORAGE = "sqlite:///optuna_simctbw"

else:
    N_CORES = 30
    N_TRIALS_PER_CORE = 5
    N_REPS_PER_MODEL = 50
    NAME_OF_RUN = "simctbw_calib"
    STUDY_NAME = NAME_OF_RUN
    STORAGE = "sqlite:///optuna_simctbw"

N_TRIALS = N_CORES * N_TRIALS_PER_CORE

run_folder_path = Path.joinpath(src.PATH, "data_output", "run_" + (utils.str_time() if NAME_OF_RUN is None else NAME_OF_RUN))


def objective(trial):
    output_dict = model_func(
        master_model_index=trial.number, 
        run_folder_path=run_folder_path, 
        output_type="save_rep_on_disk", 
        mode="calib", 
        trial=trial, 
        n_replications=N_REPS_PER_MODEL,
        )
    rmse_incidence = output_dict["rmse_incidence"]
    error = rmse_incidence
    return error

def helper_func(i):
    study = optuna.load_study(
        study_name=STUDY_NAME, 
        storage=STORAGE,
    )
    study.optimize(objective, n_trials=1)
    del(study)


if __name__ == "__main__":
    try:
        optuna.create_study(
            study_name=STUDY_NAME,  
            storage=STORAGE
            )
    except:
        pass

    with WorkerPool(n_jobs=N_CORES, use_dill=False) as pool:
        pool.map(
            helper_func, 
            range(N_TRIALS), 
            chunk_size=1,
            worker_lifespan=1,
            progress_bar=True,
            )