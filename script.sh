#!/bin/bash

#SBATCH --mail-user= raphael.degottardi@students.unibe.ch
#SBATCH --mail-type=end,fail

#SBATCH --job-name="GED with WL-hashes"
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=6
#SBATCH --time=0-01:00:00
#SBATCH --output=./slurms/test_with_preds_%A_%a.out
#SBATCH --array=1


param_store=$HOME/Info_Bachelor/BachelorThesis_MyCode/args.txt

# Get first argument
dataset=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')

# Put your code below this line
module load Python/3.9
cd $HOMEE/Info_Bachelor/graph_matching-core
source venv/bin/activate
echo "I'm on host: $HOSTNAME"

cd $HOMEE/Info_Bachelor/BachelorThesis_MyCode
srun python Protoype_script.py --dataset $dataset

