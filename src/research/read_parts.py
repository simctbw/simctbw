import src
import src.entities.utils as utils
from pathlib import Path
import shutil

run_folder_path = Path.joinpath(
    src.PATH, "data_output/run_baseline_replication2"
)
NAME_OF_RUN = "baseline_replication2"
utils.read_parts(divisor=20, run_folder_path=run_folder_path, name_of_run=NAME_OF_RUN)