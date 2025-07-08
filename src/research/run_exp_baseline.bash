#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=2:00:00
#SBATCH --mem=150gb

module load devel/miniconda/3

source $MINICONDA_HOME/etc/profile.d/conda.sh

conda run -n simctbw_revision python run_exp.py baseline baseline