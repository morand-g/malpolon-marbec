#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --job-name=malpolon_rls
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --dependency=afterany:15772
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=2
#SBATCH -o malpolon_rls.out
#SBATCH -e malpolon_rls.err

source /home/gmorand/venvs/deepsdm3/bin/activate
python rls_aus_binned.py -cn "rls_aus_binned_pa" -m run.seed=1,2,3,4,5
#nsys profile --trace=cuda,nvtx,osrt --force-overwrite true -o profiling_test python rls_aus_binned.py -cn "rls_aus_fm"