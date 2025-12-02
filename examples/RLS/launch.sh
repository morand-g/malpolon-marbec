#!/bin/bash

#SBATCH --partition=standard
#SBATCH --job-name=malpolon_rls
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --dependency=afterany:14066
#SBATCH -o malpolon_rls.out
#SBATCH -e malpolon_rls.err

source /home/gmorand/venvs/deepsdm2/bin/activate
#nsys profile --trace=cuda,nvtx,osrt -o profiling_test python rls_aus_binned.py -cn "rls_aus_binned"
python rls_aus_binned.py -cn "rls_aus_binned"