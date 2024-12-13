#!/bin/bash

#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=4          # 32 tasks per node
#SBATCH --time=24:00:00               # time limits: 1 hour
#SBATCH --error=SWE_Eval.err            # standard error file
#SBATCH --output=SWE_Eval.out           # standard output file
#SBATCH --account=IscrC_TfG      # account name
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --gres=gpu:4
#SBATCH --mem=64G

srun python test.py 