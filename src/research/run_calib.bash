#!/bin/bash
#SBATCH --job-name=simctbw_calib
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --time=8:00:00
#SBATCH --mem=200gb

module load devel/miniconda/3

source $MINICONDA_HOME/etc/profile.d/conda.sh

conda run -n soepsim python run_calib.py
