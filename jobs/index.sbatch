#!/bin/bash
#SBATCH --job-name=generate_index
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mf3734@nyu.edu
#SBATCH --output=../logs_localize_sound_dec2021/index_output_logs/slurm_%j_%x__index.log

#SBATCH --mem=96GB
#SBATCH --time=1-23:00

#########################
#
# Job: ICASSP baseline
#
#########################




##### Load Modules



##### Setup Environment






##### Run Command
cd .
# run script with arguments
singularity exec --nv --overlay /scratch/mf3734/share/urbansas/icasspBaseline-5GB-200K.ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif  bash -c ". /ext3/env.sh;  \
        python index_dataset.py --datasets URBANSAS --overwrite
	"

