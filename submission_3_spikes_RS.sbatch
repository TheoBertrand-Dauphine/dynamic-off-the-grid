#!/bin/sh
# fichier submission.SBATCH
#SBATCH --nodes=1
#SBATCH -c 1
#SBATCH --job-name="3spikestuning_euclidean"
#SBATCH --output=%x.out
#SBATCH --time=96:00:00
#SBATCH --error=%x.out
#SBATCH --mail-user=tbertrand@ceremade.dauphine.fr
#SBATCH --mail-type=BEGIN,FAIL,END

python3 -m wandb agent tbertrand/dynamic-off-the-grid/cvkrjyix --count 200