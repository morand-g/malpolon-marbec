#!/bin/bash

#SBATCH --partition=standard
#SBATCH --job-name=malpolon_rls
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --dependency=afterany:9674
#SBATCH -o malpolon_rls.out
#SBATCH -e malpolon_rls.err


source /home/gmorand/venvs/deepsdm2/bin/activate
python rls_aus_reg.py 