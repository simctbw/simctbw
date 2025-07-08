#!/bin/bash
#SBATCH --job-name=sens_iso_level_helix
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=62
#SBATCH --time=15:00:00
#SBATCH --mem=60gb
#SBATCH --mail-type=BEGIN,END,FAIL


module load devel/miniforge

source $MINICONDA_HOME/etc/profile.d/conda.sh

conda run -n simctbw_revision python run_exp.py contact_tracing sens_iso_level_helix
