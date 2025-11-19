#!/bin/bash

#SBATCH --partition=standard
#SBATCH --job-name=malpolon_rls
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
###SBATCH --begin=now+1hour
#SBATCH --dependency=afterany:13363
#SBATCH -o malpolon_rls2.out
#SBATCH -e malpolon_rls2.err


source /home/gmorand/venvs/deepsdm2/bin/activate
python rls_aus_binned.py -cn "rls_aus_binned"