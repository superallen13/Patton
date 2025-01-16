#!/bin/bash
#SBATCH --account=a_huang
#SBATCH --cpus-per-task=16
#SBATCH --error=logs/%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=patton_pretrain
#SBATCH --mem=200GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%j.out
#SBATCH --partition=gpu_sxm
#SBATCH --qos=sxm
#SBATCH --time=7-00:00:00

# Load modules
module load cuda/12.2
module load miniconda3/4.12.0
module load gcc/10
source $EBROOTMINICONDA3/etc/profile.d/conda.sh
conda activate TAGE

JOB_SCRIPT=""
while getopts ":j:" opt; do
    case $opt in
    j)
        JOB_SCRIPT="$OPTARG"
        ;;
    \?)
        echo "Invalid option: -$OPTARG" >&2
        ;;
    esac
done

echo "Will execute: bash ${JOB_SCRIPT}.sh"

bash "${JOB_SCRIPT}.sh"
