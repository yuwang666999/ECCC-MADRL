#!/bin/sh
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --job-name=ECCC_MADRL_MEC
#SBATCH --time 60:00:00
source activate vECCCMADRLMEC
python run.py $SLURM_ARRAY_TASK_ID