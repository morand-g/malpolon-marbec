#!/bin/bash

#SBATCH --partition=standard
#SBATCH --job-name=malpolon_rls_reg
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --dependency=afterany:15803
#SBATCH --cpus-per-task=4
#SBATCH -o malpolon_rls_reg.out
#SBATCH -e malpolon_rls_reg.err

source /home/gmorand/venvs/deepsdm2/bin/activate
python rls_aus_reg.py