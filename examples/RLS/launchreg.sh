#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --job-name=malpolon_rls_reg
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --dependency=afterany:16022
#SBATCH --ntasks-per-node=4
#SBATCH -o malpolon_rls_reg.out
#SBATCH -e malpolon_rls_reg.err

source /home/gmorand/venvs/deepsdm2/bin/activate
python rls_aus_reg.py -cn "rls_aus_reg_eco"