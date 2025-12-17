#!/bin/bash

#SBATCH --partition=standard
#SBATCH --job-name=malpolon_rls
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --dependency=afterany:14947
#SBATCH --cpus-per-task=4
#SBATCH -o malpolon_rls.out
#SBATCH -e malpolon_rls.err

source /home/gmorand/venvs/deepsdm2/bin/activate
python rls_aus_binned.py -cn "rls_aus_binned"
#nsys profile --trace=cuda,nvtx,osrt --force-overwrite true -o profiling_test python rls_aus_binned.py -cn "rls_aus_fm"