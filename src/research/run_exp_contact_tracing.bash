#!/bin/bash
#SBATCH --job-name=contact_tracing
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=5:00:00
#SBATCH --mem=250gb

module load devel/miniconda/3

source $MINICONDA_HOME/etc/profile.d/conda.sh

conda run -n simctbw python run_exp.py contact_tracing ct_exp
