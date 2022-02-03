#!/bin/bash
#SBATCH -c 2                               # Request # cores
#SBATCH --mem=20G                          # Memory total in MB (for all cores)
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH -t 0-8:59                        # Runtime in D-HH:MM format

#SBATCH -p gpu                         # Partition to run in
#SBATCH -N 1                            # Request one node (if you request more than one core with -c, also using
#SBATCH -o slurm_outputs/slurm-job_%j--array-ind_%a.out                 # File to which STDOUT + STDERR will be written, including job ID in filename

hostname
pwd
module load  gcc/6.2.0 cuda/10.2

srun /home/ald339/.conda/envs/sdm_env/bin/python \
  slurm_runner.py --exp_ind ${SLURM_ARRAY_TASK_ID} --num_workers ${SLURM_CPUS_PER_TASK} --total_tasks ${SLURM_ARRAY_TASK_MAX}